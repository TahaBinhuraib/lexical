import datetime
import json
import os
import random
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.utils.data as torch_data
from absl import app, flags
from nltk.translate.bleu_score import corpus_bleu
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import hlog
import utils.myutil as myutil
from data import collate, encode, encode_io, eval_format, get_fig2_exp
from mutex import MultiIter, Mutex, RecordLoss, Vocab
from src import NoamLR, SoftAlign
from utils.make_data import generate_data

sns.set()
FLAGS = flags.FLAGS
flags.DEFINE_string("language_task", "morphology", "folder to find data")
flags.DEFINE_string("language", "kongo", "low resource language")
flags.DEFINE_integer("dim", 512, "trasnformer dimension")
flags.DEFINE_integer("n_layers", 2, "number of rnn layers")
flags.DEFINE_integer("n_batch", 64, "batch size")
flags.DEFINE_float("gclip", 0.5, "gradient clip")
flags.DEFINE_integer("n_epochs", 2, "number of training epochs")
flags.DEFINE_integer("beam_size", 5, "beam search size")
flags.DEFINE_float("lr", 1.0, "learning rate")
flags.DEFINE_float("temp", 1.0, "temperature for samplings")
flags.DEFINE_float("dropout", 0.4, "dropout")
flags.DEFINE_string("save_model", "model.m", "model save location")
flags.DEFINE_string("load_model", "", "load pretrained model")
flags.DEFINE_integer("seed", 0, "random seed")
flags.DEFINE_bool("debug", False, "debug mode")
flags.DEFINE_bool("regularize", False, "regularization")
flags.DEFINE_bool("bidirectional", False, "bidirectional encoders")
flags.DEFINE_bool("attention", True, "Source Attention")
flags.DEFINE_integer("warmup_steps", 4000, "noam warmup_steps")
flags.DEFINE_integer("valid_steps", 500, "validation steps")
flags.DEFINE_integer("max_step", 8000, "maximum number of steps")
flags.DEFINE_integer("tolarance", 1, "early stopping tolarance")
flags.DEFINE_integer("accum_count", 4, "grad accumulation count")
flags.DEFINE_bool("shuffle", True, "shuffle training set")
flags.DEFINE_bool("lr_schedule", True, "noam lr scheduler")
flags.DEFINE_bool("qxy", True, "train pretrained qxy")
flags.DEFINE_bool("copy", True, "enable copy mechanism")
flags.DEFINE_bool("highdrop", False, "high drop mechanism")
flags.DEFINE_bool("highdroptest", False, "high drop at test")
flags.DEFINE_float("highdropvalue", 0.5, "high drop value")
flags.DEFINE_string("aligner", "", "alignment file by fastalign")
flags.DEFINE_bool("soft_align", False, "lexicon projection matrix")
flags.DEFINE_bool("learn_align", False, "learned lexicon projection matrix")
# flags.DEFINE_float("paug", 0.1, "augmentation ratio")
flags.DEFINE_string("aug_file", "", "data source for augmentation")
flags.DEFINE_float("soft_temp", 0.2, "2 * temperature for soft lexicon")
flags.DEFINE_string("tb_dir", "", "tb_dir")
plt.rcParams["figure.dpi"] = 300

ROOT_FOLDER = os.path.dirname(os.path.realpath(__file__))
DEVICE = torch.device(("cuda" if torch.cuda.is_available() else "cpu"))


def read_augmented_file(file, vocab_x, vocab_y):
    with open(file, "r") as f:
        data = json.load(f)
    edata = []
    for datum in data:
        inp, out = datum["inp"], datum["out"]
        edata.append(encode_io((inp, out), vocab_x, vocab_y))
    return edata


def train(model, train_dataset, val_dataset, writer=None, references=None):
    # opt = Adafactor(
    # model.parameters(), scale_parameter=True, relative_step=True, warmup_init=True, lr=FLAGS.lr
    # )
    opt = optim.Adam(model.parameters(), lr=FLAGS.lr, betas=(0.9, 0.998))

    if FLAGS.lr_schedule:
        # scheduler = AdafactorSchedule(opt)
        scheduler = NoamLR(opt, FLAGS.dim, warmup_steps=FLAGS.warmup_steps)
    else:
        scheduler = None

    if FLAGS.aug_file != "":
        aug_data = read_augmented_file(FLAGS.aug_file, model.vocab_x, model.vocab_y)
        random.shuffle(train_dataset)
        random.shuffle(aug_data)
        titer = MultiIter(train_dataset, aug_data, 1 - FLAGS.paug)
        train_loader = torch_data.DataLoader(
            titer, batch_size=FLAGS.n_batch, shuffle=False, collate_fn=collate
        )
    else:
        train_loader = torch_data.DataLoader(
            train_dataset, batch_size=FLAGS.n_batch, shuffle=FLAGS.shuffle, collate_fn=collate
        )

    tolarance = FLAGS.tolarance
    best_f1 = best_acc = -np.inf
    best_loss = np.inf
    best_bleu = steps = accum_steps = 0
    got_nan = False
    is_running = lambda: not got_nan and accum_steps < FLAGS.max_step and tolarance > 0
    while is_running():
        train_loss = train_batches = 0
        opt.zero_grad()
        recorder = RecordLoss()
        for inp, out, lens in tqdm(train_loader):
            if not is_running():
                break
            nll = model(inp.to(DEVICE), out.to(DEVICE), lens=lens.to(DEVICE), recorder=recorder)
            steps += 1
            loss = nll / FLAGS.accum_count
            loss.backward()
            print(f"step:: {steps}")
            train_loss += loss.detach().item() * FLAGS.accum_count
            train_batches += 1
            if steps % FLAGS.accum_count == 0:
                accum_steps += 1
                gnorm = nn.utils.clip_grad_norm_(model.parameters(), FLAGS.gclip)
                if not np.isfinite(gnorm):
                    got_nan = True
                    print("=====GOT NAN=====")
                    break
                opt.step()
                opt.zero_grad()

                if scheduler is not None:
                    scheduler.step()

                if accum_steps % FLAGS.valid_steps == 0:
                    with hlog.task(accum_steps):
                        hlog.value("curr loss", train_loss / train_batches)
                        acc, f1, val_loss, bscore = validate(
                            model, val_dataset, writer=writer, references=references
                        )
                        model.train()
                        hlog.value("acc", acc)
                        hlog.value("f1", f1)
                        hlog.value("bscore", bscore)
                        best_f1 = max(best_f1, f1)
                        best_acc = max(best_acc, acc)
                        best_bleu = max(best_bleu, bscore)
                        hlog.value("val_loss", val_loss)
                        hlog.value("best_acc", best_acc)
                        hlog.value("best_f1", best_f1)
                        hlog.value("best_bleu", best_bleu)
                        if val_loss < best_loss:
                            best_loss = val_loss
                            tolarance = FLAGS.tolarance
                        else:
                            tolarance -= 1
                        hlog.value("best_loss", best_loss)

    hlog.value("final_acc", acc)
    hlog.value("final_f1", f1)
    hlog.value("final_bleu", bscore)
    hlog.value("best_acc", best_acc)
    hlog.value("best_f1", best_f1)
    hlog.value("best_loss", best_loss)
    hlog.value("best_bleu", best_bleu)
    return acc, f1, bscore


def validate(model, val_dataset, vis=False, final=False, writer=None, references=None):
    model.eval()
    val_loader = torch_data.DataLoader(
        val_dataset, batch_size=FLAGS.n_batch, shuffle=False, collate_fn=collate
    )
    total = correct = loss = tp = fp = fn = 0
    acc_list = []
    cur_references = []
    candidates = []
    with torch.no_grad():
        for inp, out, lens in tqdm(val_loader):
            input = inp.to(DEVICE)
            lengths = lens.to(DEVICE)
            pred, _ = model.sample(
                input,
                lens=lengths,
                temp=1.0,
                max_len=model.MAXLEN_Y,
                greedy=True,
                beam_size=FLAGS.beam_size * final,
                calc_score=False,
            )

            loss += model.pyx(input, out.to(DEVICE), lens=lengths).item() * input.shape[1]
            for i, seq in enumerate(pred):
                true_char = 0
                ref = out[:, i].numpy().tolist()
                ref = eval_format(model.vocab_y, ref)
                pred_here = eval_format(model.vocab_y, pred[i])
                if references is None:
                    cur_references.append([ref])
                else:
                    inpref = " ".join(model.vocab_x.decode(inp[0 : lens[i], i].numpy().tolist()))
                    cur_references.append(references[inpref])

                len_ref = len(ref)
                len_pred_here = len(pred_here)
                N = max(len_ref, len_pred_here)

                for y, y_hat in zip(ref, pred_here):
                    if y == y_hat:
                        true_char += 1

                acc_here = true_char / N
                acc_list.append(acc_here)

                candidates.append(pred_here)
                correct_here = pred_here == ref
                correct += correct_here
                tp_here = len([p for p in pred_here if p in ref])
                tp += tp_here
                fp_here = len([p for p in pred_here if p not in ref])
                fp += fp_here
                fn_here = len([p for p in ref if p not in pred_here])
                fn += fn_here
                total += 1
                if vis:
                    with hlog.task(total):
                        hlog.value("label", correct_here)
                        hlog.value("tp", tp_here)
                        hlog.value("fp", fp_here)
                        hlog.value("fn", fn_here)
                        inp_lst = inp[:, i].detach().cpu().numpy().tolist()
                        hlog.value("input", eval_format(model.vocab_x, inp_lst))
                        hlog.value("gold", ref)
                        hlog.value("pred", pred_here)

    if writer is not None:
        writer.add_scalar(f"Loss/eval/loss", loss / total)
        writer.add_scalar(f"Loss/eval/character_acc", mean(acc_list))
        writer.add_scalar(f"Loss/eval/accuracy", correct / total)
        writer.flush()

    bleu_score = corpus_bleu(cur_references, candidates)
    acc = correct / total
    loss = loss / total
    if tp + fp > 0:
        prec = tp / (tp + fp)
    else:
        prec = 0
    rec = tp / (tp + fn)
    if prec == 0 or rec == 0:
        f1 = 0
    else:
        f1 = 2 * prec * rec / (prec + rec)
    hlog.value("acc", acc)
    hlog.value("f1", f1)
    hlog.value("bleu", bleu_score)
    hlog.value("acc_char", mean(acc_list))
    return acc, f1, loss, bleu_score


def tranlate_with_alignerv2(aligner, vocab_x, vocab_y, unwanted=lambda x: False, temp=0.02):
    if aligner == "uniform":
        proj = np.ones((len(vocab_x), len(vocab_y)), dtype=np.float32)
    elif aligner == "random":
        proj = np.random.default_rng().random((len(vocab_x), len(vocab_y)), dtype=np.float32)
    else:
        proj = np.zeros((len(vocab_x), len(vocab_y)), dtype=np.float32)

        x_keys = list(vocab_x._contents.keys())
        y_keys = list(vocab_y._contents.keys())

        for (x, x_key) in enumerate(x_keys):
            if x_key in y_keys:
                y = vocab_y[x_key]
                proj[x, y] = 1.0

        with open(aligner, "r") as handle:
            word_alignment = json.load(handle)
            for (w, a) in word_alignment.items():
                if w in vocab_x and len(a) > 0 and not unwanted(w):
                    x = vocab_x[w]
                    for (v, n) in a.items():
                        if not unwanted(v) and v in vocab_y:
                            y = vocab_y[v]
                            proj[x, y] = 2 * n

        empty_xs = np.where(proj.sum(axis=1) == 0)[0]
        empty_ys = np.where(proj.sum(axis=0) == 0)[0]

        if len(empty_ys) != 0 and len(empty_xs) != 0:
            for i in empty_xs:
                proj[i, empty_ys] = 1 / len(empty_ys)

    if FLAGS.soft_align:
        return SoftAlign(proj / FLAGS.soft_temp, requires_grad=FLAGS.learn_align).to(DEVICE)
    else:
        return np.argmax(proj, axis=1)


def tranlate_with_aligner(aligner, vocab_x, vocab_y, unwanted=lambda x: False, temp=0.02):
    proj = np.identity(len(vocab_x), dtype=np.float32)
    vocab_keys = list(vocab_x._contents.keys())
    with open(aligner, "r") as handle:
        word_alignment = json.load(handle)
        for (w, a) in word_alignment.items():
            if w in vocab_x and len(a) > 0 and not unwanted(w):
                x = vocab_keys.index(w)
                for (v, n) in a.items():
                    if not unwanted(v) and v in vocab_y:
                        y = vocab_keys.index(v)
                        proj[x, y] = 2 * n

    if FLAGS.soft_align:
        return SoftAlign(proj / temp, requires_grad=FLAGS.learn_align).to(DEVICE)
    else:
        return np.argmax(proj, axis=1)


def copy_translation_cogs(vocab_x, vocab_y):
    if FLAGS.aligner != "":
        return tranlate_with_alignerv2(FLAGS.aligner, vocab_x, vocab_y)
    else:
        proj = np.zeros((len(vocab_x), len(vocab_y)), dtype=np.float32)
        x_keys = list(vocab_x._contents.keys())
        y_keys = list(vocab_y._contents.keys())
        for (x, x_key) in enumerate(x_keys):
            if x_key in y_keys:
                y = y_keys.index(x_key)
                proj[x, y] = 1.0

        return np.argmax(proj, axis=1)


def copy_translation_mutex(vocab_x, vocab_y, primitives):
    if FLAGS.aligner != "":
        return tranlate_with_alignerv2(FLAGS.aligner, vocab_x, vocab_y)
    else:
        proj = np.identity(len(vocab_x))
        vocab_keys = list(vocab_x._contents.keys())
        for (x, y) in primitives:
            idx = vocab_keys.index(x[0])
            idy = vocab_keys.index(y[0])
            print("x: ", x[0], " y: ", y[0])
            proj[idx, idx] = 0
            proj[idx, idy] = 1
        return np.argmax(proj, axis=1)


def copy_translation_scan(vocab_x, vocab_y):
    if FLAGS.aligner != "":
        return tranlate_with_alignerv2(FLAGS.aligner, vocab_x, vocab_y)
    else:
        proj = np.identity(len(vocab_x))
        vocab_keys = list(vocab_x._contents.keys())
        for (x, y) in [
            ("jump", "I_JUMP"),
            ("walk", "I_WALK"),
            ("look", "I_LOOK"),
            ("run", "I_RUN"),
            ("right", "I_TURN_RIGHT"),
            ("left", "I_TURN_LEFT"),
        ]:

            idx = vocab_keys.index(x)
            idy = vocab_keys.index(y)
            print("x: ", x, " y: ", y, "idx: ", idx, "idy: ", idy)
            proj[idx, idx] = 0
            proj[idx, idy] = 1
        return np.argmax(proj, axis=1)


def copy_translation_translate(vocab_x, vocab_y):
    if FLAGS.aligner != "":
        return tranlate_with_alignerv2(FLAGS.aligner, vocab_x, vocab_y)
    else:
        proj = np.zeros((len(vocab_x), len(vocab_y)), dtype=np.float32)
        x_keys = list(vocab_x._contents.keys())
        y_keys = list(vocab_y._contents.keys())
        for (x, x_key) in enumerate(x_keys):
            if x_key in y_keys:
                y = y_keys.index(x_key)
                proj[x, y] = 1.0
        return np.argmax(proj, axis=1)


def copy_translation_cfq(vocab_x, vocab_y):
    assert FLAGS.aligner != "", "Predefined aligner is needed for translate exps"
    return tranlate_with_alignerv2(FLAGS.aligner, vocab_x, vocab_y)


def main(argv):
    hlog.flags()
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)

    vocab_x = Vocab()
    vocab_y = Vocab()
    references = None

    if FLAGS.language_task == "morphology":
        train_path = f"data_2020/{FLAGS.language}/{FLAGS.language}.trn"
        dev_path = f"data_2020/{FLAGS.language}/{FLAGS.language}.dev"
        test_path = f"data_2020/{FLAGS.language}/{FLAGS.language}.tst"

        train_input, train_output, train_tags = myutil.read_data(train_path)
        validate_input, validate_output, validate_tags = myutil.read_data(dev_path)
        test_input, test_tags = myutil.read_test_data(test_path)

        characters = myutil.get_chars(
            train_input + train_output + validate_input + validate_output
        )
        tags = myutil.get_tags(train_tags + validate_tags)

        if FLAGS.copy:
            for tag in tags:
                vocab_x.add(tag)
                vocab_y.add(tag)
        else:
            for tag in tags:
                vocab_x.add(tag)

        for c in characters:
            vocab_x.add(c)
            vocab_y.add(c)

        train_items, test_items, val_items, study, test = generate_data(
            train_input,
            train_tags,
            validate_input,
            validate_tags,
            train_output,
            validate_output,
            vocab_x,
            vocab_y,
        )

        # How to determine this length?
        max_len_x = 12
        max_len_y = 12

    if FLAGS.copy:
        # vocab_y = vocab_x.merge(vocab_y)
        copy_translation = copy_translation_mutex(
            vocab_x, vocab_y, study[0:4]
        )  # FIXME: make sure they are primitives
    else:
        copy_translation = None

    hlog.value("vocab_x\n", vocab_x)
    hlog.value("vocab_y\n", vocab_y)
    hlog.value("study\n", study)
    hlog.value("test\n", test)

    writer = None
    if FLAGS.tb_dir != "":
        tb_log_dir = (
            FLAGS.tb_dir
            + f"/seed_{FLAGS.seed}_"
            + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        )
        writer = SummaryWriter(log_dir=tb_log_dir)
    else:
        writer = None

    if FLAGS.load_model == "":
        model = Mutex(
            vocab_x,
            vocab_y,
            FLAGS.dim,
            FLAGS.dim,
            max_len_x=max_len_x,
            max_len_y=max_len_y,
            copy=FLAGS.copy,
            n_layers=FLAGS.n_layers,
            self_att=False,
            attention=FLAGS.attention,
            dropout=FLAGS.dropout,
            temp=FLAGS.temp,
            qxy=FLAGS.qxy,
            bidirectional=FLAGS.bidirectional,  # TODO remember human data was bidirectional
        ).to(DEVICE)
        if copy_translation is not None:
            model.pyx.decoder.copy_translation = copy_translation

        with hlog.task("train model"):
            acc, f1, bscore = train(
                model, train_items, val_items, writer=writer, references=references
            )
    else:
        model = torch.load(FLAGS.load_model)

    with hlog.task("train evaluation"):
        validate(model, train_items, vis=False, references=references)

    with hlog.task("val evaluation"):
        validate(model, val_items, vis=True, references=references)

    with hlog.task("test evaluation (greedy)"):
        validate(model, test_items, vis=True, final=False, references=references)

    # with hlog.task("test evaluation (beam)"):
    #     validate(model, test_items, vis=False, final=True)

    # torch.save(model, f"seed_{FLAGS.seed}_" + FLAGS.save_model)


if __name__ == "__main__":
    app.run(main)
