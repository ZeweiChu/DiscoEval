import torch
import config
import train_helper
import data_utils
import sep_dec_models as models

import numpy as np

from tensorboardX import SummaryWriter

MAX_NSENT = 164  # maximum number of sentences in a paragraph
MAX_NPARA = 5876  # maximum number of paragraphs
MAX_NLV = 7  # maximum number of levels

test_bm_res = test_avg_res = 0


def run(e):
    global test_bm_res, test_avg_res

    e.log.info("*" * 25 + " DATA PREPARATION " + "*" * 25)
    dp = data_utils.data_processor(
        train_path=e.config.train_path,
        eval_path=e.config.eval_path,
        experiment=e)
    data, W = dp.process()

    e.log.info("*" * 25 + " DATA PREPARATION " + "*" * 25)
    e.log.info("*" * 25 + " MODEL INITIALIZATION " + "*" * 25)

    if e.config.model.lower() == "basic":
        model_class = models.basic_model
    elif e.config.model.lower() == "pos":
        model_class = models.pos_model
    elif e.config.model.lower() == "quant_pos":
        model_class = models.quantize_pos_model
    elif e.config.model.lower() == "quant_pos_reg":
        model_class = models.quantize_pos_regression_model
    elif e.config.model.lower() == "quant_attn_pos1":
        model_class = models.quantize_attn_pos_model1
    elif e.config.model.lower() == "quant_attn_pos2":
        model_class = models.quantize_attn_pos_model2

    model = model_class(
        vocab_size=len(data.sent_vocab),
        doc_title_vocab_size=len(data.doc_title_vocab),
        sec_title_vocab_size=len(data.sec_title_vocab),
        embed_dim=e.config.edim if W is None else W.shape[1],
        embed_init=W,
        max_nsent=MAX_NSENT,
        max_npara=MAX_NPARA,
        max_nlv=MAX_NLV,
        experiment=e)

    start_epoch = it = n_finish_file = 0
    todo_file = list(range(len(data.train_data)))

    if e.config.resume:
        start_epoch, it, test_bm_res, test_avg_res, todo_file = \
            model.load()
        if e.config.use_cuda:
            model.cuda()
            e.log.info("transferred model to gpu")
        e.log.info(
            "resumed from previous checkpoint: start epoch: {}, "
            "iteration: {}, test benchmark {:.3f}, test avg res: {:.3f}"
            .format(start_epoch, it, test_bm_res, test_avg_res))

    e.log.info(model)
    e.log.info("*" * 25 + " MODEL INITIALIZATION " + "*" * 25)

    if e.config.summarize:
        writer = SummaryWriter(e.experiment_dir)

    evaluator = train_helper.evaluator(model, e)

    e.log.info("Training start ...")
    train_stats = train_helper.tracker(
        ["loss", "prev_logloss", "next_logloss", "para_loss", "sent_loss",
         "level_loss", "doc_title_loss", "sec_title_loss"])

    for epoch in range(start_epoch, e.config.n_epoch):
        while len(todo_file):
            file_idx = np.random.randint(len(todo_file))
            train_file = data.train_data[todo_file[file_idx]]

            with data_utils.minibatcher(
                    train_file=train_file,
                    sent_vocab=data.sent_vocab,
                    doc_title_vocab=data.doc_title_vocab,
                    sec_title_vocab=data.sec_title_vocab,
                    batch_size=e.config.batch_size,
                    max_len=e.config.max_len,
                    max_nsent=MAX_NSENT,
                    max_npara=MAX_NPARA,
                    bow=e.config.decoder_type.lower() == "bag_of_words",
                    log=e.log) as train_batch:

                for doc_id, para_id, pmask, _, sent_id, \
                        smask, lv, s, m, t, tm, t2, tm2, dt, st in \
                        train_batch:
                    it += 1

                    loss, logloss1, logloss2, para_loss, sent_loss, \
                        level_loss, doc_title_loss, sec_title_loss = model(
                            s, m, t, tm, t2, tm2, doc_id, para_id,
                            pmask, sent_id, smask, lv, dt, st)

                    model.optimize(loss)

                    train_stats.update(
                        {"loss": loss, "prev_logloss": logloss1,
                         "next_logloss": logloss2, "para_loss": para_loss,
                         "sent_loss": sent_loss, "level_loss": level_loss,
                         "doc_title_loss": doc_title_loss,
                         "sec_title_loss": sec_title_loss},
                        len(s))

                    if it % e.config.print_every == 0:
                        summarization = train_stats.summarize(
                            "epoch: {}, it: {}".format(epoch, it))
                        e.log.info(summarization)
                        if e.config.summarize:
                            for name, value in train_stats.stats.items():
                                writer.add_scalar(
                                    "train/" + name, value, it)
                        train_stats.reset()

                    if it % e.config.eval_every == 0:

                        e.log.info("*" * 25 + " STS EVAL " + "*" * 25)

                        test_stats, test_bm_res, test_avg_res, test_avg_s = \
                            evaluator.evaluate(data.test_data, 'score_sts')

                        e.log.info("*" * 25 + " STS EVAL " + "*" * 25)

                        # model.save(
                        #     test_avg=test_avg_res,
                        #     test_bm=test_bm_res,
                        #     todo_file=train_batch.todo_file,
                        #     it=it,
                        #     epoch=epoch)

                        if e.config.summarize:
                            for year, stats in test_stats.items():
                                writer.add_scalar(
                                    "test/{}_pearson".format(year),
                                    stats[1], it)
                                writer.add_scalar(
                                    "test/{}_spearman".format(year),
                                    stats[2], it)

                            writer.add_scalar(
                                "test/avg_pearson", test_avg_res, it)
                            writer.add_scalar(
                                "test/avg_spearman", test_avg_s, it)
                            writer.add_scalar(
                                "test/STSBenchmark_pearson", test_bm_res, it)

                        e.log.info("STSBenchmark result: {:.4f}, "
                                   "test average result: {:.4f}"
                                   .format(test_bm_res, test_avg_res))

            del todo_file[file_idx]

            n_finish_file += 1
            model.save(
                test_avg=test_avg_res,
                test_bm=test_bm_res,
                todo_file=todo_file,
                it=it,
                epoch=epoch if len(todo_file) else epoch + 1)

            time_per_file = e.elapsed_time / n_finish_file
            time_in_need = time_per_file * (e.config.n_epoch - epoch - 1) * \
                len(data.train_data) + time_per_file * len(todo_file)
            e.log.info("elapsed time: {:.2f}(h), "
                       "#finished file: {}, #todo file: {}, "
                       "time per file: {:.2f}(h), "
                       "time needed to finish: {:.2f}(h)"
                       .format(e.elapsed_time, n_finish_file, len(todo_file),
                               time_per_file, time_in_need))

            if time_per_file + e.elapsed_time > 3.8 \
                    and e.config.auto_disconnect:
                exit(1)

        todo_file = list(range(len(data.train_data)))


if __name__ == '__main__':

    args = config.get_base_parser().parse_args()
    args.use_cuda = torch.cuda.is_available()

    def exit_handler(*args):
        print(args)
        print("STSBenchmark result: {:.4f}, "
              "test average result: {:.4f}"
              .format(test_bm_res, test_avg_res))
        exit()

    train_helper.register_exit_handler(exit_handler)

    with train_helper.experiment(args, args.save_prefix) as e:

        e.log.info("*" * 25 + " ARGS " + "*" * 25)
        e.log.info(args)
        e.log.info("*" * 25 + " ARGS " + "*" * 25)

        run(e)
