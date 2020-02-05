# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer speech recognition model (pytorch)."""

from argparse import Namespace
from distutils.util import strtobool

import logging
import math
import numpy as np

import torch

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD
from espnet.nets.pytorch_backend.e2e_asr import Reporter
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.add_sos_eos import factorize_predict
from espnet.nets.pytorch_backend.transformer.add_sos_eos import mask_predict
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.transformer.plot import PlotAttentionReport
from espnet.nets.scorers.ctc import CTCPrefixScorer


class E2E(ASRInterface, torch.nn.Module):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        group = parser.add_argument_group("transformer model setting")

        group.add_argument("--transformer-init", type=str, default="pytorch",
                           choices=["pytorch", "xavier_uniform", "xavier_normal",
                                    "kaiming_uniform", "kaiming_normal"],
                           help='how to initialize transformer parameters')
        group.add_argument("--transformer-input-layer", type=str, default="conv2d",
                           choices=["conv2d", "linear", "embed"],
                           help='transformer input layer type')
        group.add_argument('--transformer-attn-dropout-rate', default=None, type=float,
                           help='dropout in transformer attention. use --dropout-rate if None is set')
        group.add_argument('--transformer-lr', default=10.0, type=float,
                           help='Initial value of learning rate')
        group.add_argument('--transformer-warmup-steps', default=25000, type=int,
                           help='optimizer warmup steps')
        group.add_argument('--transformer-length-normalized-loss', default=True, type=strtobool,
                           help='normalize loss by length')

        group.add_argument('--dropout-rate', default=0.0, type=float,
                           help='Dropout rate for the encoder')
        # Encoder
        group.add_argument('--elayers', default=4, type=int,
                           help='Number of encoder layers (for shared recognition part in multi-speaker asr mode)')
        group.add_argument('--eunits', '-u', default=300, type=int,
                           help='Number of encoder hidden units')
        # Attention
        group.add_argument('--adim', default=320, type=int,
                           help='Number of attention transformation dimensions')
        group.add_argument('--aheads', default=4, type=int,
                           help='Number of heads for multi head attention')
        # Decoder
        group.add_argument('--dlayers', default=1, type=int,
                           help='Number of decoder layers')
        group.add_argument('--dunits', default=320, type=int,
                           help='Number of decoder hidden units')
        group.add_argument('--nat', default=False, type=strtobool,
                           help='Using non-autoregressive model')
        group.add_argument('--nat_mode', default='CM', type=str, choices=["CM", "FM"],
                           help='CM: A-CMLM; FM: A-FMLM')
        return parser

    @property
    def attention_plot_class(self):
        return PlotAttentionReport

    def __init__(self, idim, odim, args, ignore_id=-1, asr_model=None, mt_model=None):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        torch.nn.Module.__init__(self)
        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate
        self.encoder = Encoder(
            idim=idim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            input_layer=args.transformer_input_layer,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate
        )
        self.decoder = Decoder(
            odim=odim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.dunits,
            num_blocks=args.dlayers,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            self_attention_dropout_rate=args.transformer_attn_dropout_rate,
            src_attention_dropout_rate=args.transformer_attn_dropout_rate
        )
        if 'nat' in args and args.nat:
            self.mask = odim - 1
            self.sos = odim - 2
            self.eos = odim - 2
        else:
            self.sos = odim - 1
            self.eos = odim - 1
        self.odim = odim
        self.ignore_id = ignore_id
        self.subsample = [1]
        self.reporter = Reporter()

        # self.lsm_weight = a
        self.criterion = LabelSmoothingLoss(self.odim, self.ignore_id, args.lsm_weight,
                                            args.transformer_length_normalized_loss)
        # self.verbose = args.verbose
        self.reset_parameters(args)
        self.adim = args.adim
        self.mtlalpha = args.mtlalpha
        if args.mtlalpha > 0.0:
            self.ctc = CTC(odim, args.adim, args.dropout_rate, ctc_type=args.ctc_type, reduce=True)
        else:
            self.ctc = None

        if args.report_cer or args.report_wer:
            from espnet.nets.e2e_asr_common import ErrorCalculator
            self.error_calculator = ErrorCalculator(args.char_list,
                                                    args.sym_space, args.sym_blank,
                                                    args.report_cer, args.report_wer)
        else:
            self.error_calculator = None
        self.rnnlm = None
        self.nat = args.nat if 'nat' in args else False
        if self.nat:
            self.nat_mode = args.nat_mode if 'nat_mode' in args else False

    def reset_parameters(self, args):
        # initialize parameters
        initialize(self, args.transformer_init)

    def forward(self, xs_pad, ilens, ys_pad, ys_pad_asr=None):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        :return: ctc loass value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        """
        # 1. forward encoder
        xs_pad = xs_pad[:, :max(ilens)]  # for data parallel
        src_mask = (~make_pad_mask(ilens.tolist())).to(xs_pad.device).unsqueeze(-2)
        hs_pad, hs_mask = self.encoder(xs_pad, src_mask)
        
        # TODO(karita) show predicted text
        # TODO(karita) calculate these stats
        cer_ctc = None
        if self.mtlalpha == 0.0:
            loss_ctc = None
        else:
            batch_size = xs_pad.size(0)
            hs_len = hs_mask.view(batch_size, -1).sum(1)
            loss_ctc = self.ctc(hs_pad.view(batch_size, -1, self.adim), hs_len, ys_pad)
            if self.error_calculator is not None:
                ys_hat = self.ctc.argmax(hs_pad.view(batch_size, -1, self.adim)).data
                cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        
        self.hs_pad = hs_pad

        # 2. forward decoder
        if not self.nat or self.nat_mode == 'CM':
            if not self.nat:
                ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
            elif self.nat_mode == 'CM':
                ys_in_pad, ys_out_pad = mask_predict(ys_pad, self.mask, self.eos, self.ignore_id, training=self.training)
            ys_mask = target_mask(ys_in_pad, self.ignore_id, use_all=self.nat)
            pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)
        elif self.nat_mode == 'FM':
            ys_in_pad, ys_out_pad = factorize_predict(ys_pad, self.mask, self.eos, self.ignore_id)
            ys_mask = target_mask(ys_in_pad, self.ignore_id, use_all=self.nat)
            pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)

            #second iteration
            ys_in_pad, (mask1, mask2) = factorize_predict(ys_pad, self.mask, self.eos, self.ignore_id, pred_pad)
            ys_mask = target_mask(ys_in_pad, self.ignore_id, use_all=self.nat)
            pred_pad2, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)

            pred_pad = mask1.unsqueeze(-1).float() * pred_pad + mask2.unsqueeze(-1).float() * pred_pad2
        self.pred_pad = pred_pad

        # 3. compute attention loss
        loss_att = self.criterion(pred_pad, ys_out_pad)
        self.acc = th_accuracy(pred_pad.view(-1, self.odim), ys_out_pad,
                               ignore_label=self.ignore_id)


        # 5. compute cer/wer
        if self.training or self.error_calculator is None:
            cer, wer = None, None
        else:
            ys_hat = pred_pad.argmax(dim=-1)
            cer, wer = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        # copyied from e2e_asr
        alpha = self.mtlalpha
        if alpha == 0:
            self.loss = loss_att
            loss_att_data = float(loss_att)
            loss_ctc_data = None
        elif alpha == 1:
            self.loss = loss_ctc
            loss_att_data = None
            loss_ctc_data = float(loss_ctc)
        else:
            self.loss = alpha * loss_ctc + (1 - alpha) * loss_att
            loss_att_data = float(loss_att)
            loss_ctc_data = float(loss_ctc)

        loss_data = float(self.loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(loss_ctc_data, loss_att_data, self.acc, cer_ctc, cer, wer, loss_data)
        else:
            logging.warning('loss (=%f) is not correct', loss_data)
        return self.loss

    def scorers(self):
        """Scorers."""
        return dict(decoder=self.decoder, ctc=CTCPrefixScorer(self.ctc, self.eos))

    def encode(self, feat):
        """Encode acoustic features."""
        self.eval()
        feat = torch.as_tensor(feat).unsqueeze(0)
        enc_output, _ = self.encoder(feat, None)
        return enc_output.squeeze(0)

    def beamsearch(self, enc_output, recog_args, ret=None, char_list=None, rnnlm=None, use_jit=False):
        if recog_args.ctc_weight > 0.0:
            lpz = self.ctc.log_softmax(enc_output)
            lpz = lpz.squeeze(0)
        else:
            lpz = None

        h = enc_output.squeeze(0)

        logging.info('input lengths: ' + str(h.size(0)))
        # search parms
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight

        # preprare sos
        y = self.sos
        vy = h.new_zeros(1).long()

        if recog_args.maxlenratio == 0:
            maxlen = h.shape[0]
        else:
            # maxlen >= 1
            maxlen = max(1, int(recog_args.maxlenratio * h.size(0)))
        if self.nat and ret is not None:
            maxlen = min(maxlen, ret.size(1))
        minlen = int(recog_args.minlenratio * h.size(0))
        if self.nat and ret is not None:
            minlen = min(minlen, ret.size(1))
        logging.info('max output length: ' + str(maxlen))
        logging.info('min output length: ' + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {'score': 0.0, 'yseq': [y], 'rnnlm_prev': None, 'local_score':[]}
        else:
            hyp = {'score': 0.0, 'yseq': [y], 'local_score':[]}

        start = 0
        if lpz is not None:
            import numpy

            from espnet.nets.ctc_prefix_score import CTCPrefixScore

            ctc_prefix_score = CTCPrefixScore(lpz.detach().numpy(), 0, self.eos, numpy)
            hyp['ctc_state_prev'] = ctc_prefix_score.initial_state()
            hyp['ctc_score_prev'] = 0.0
            if ctc_weight != 1.0:
                # pre-pruning based on attention scores
                from espnet.nets.pytorch_backend.rnn.decoders import CTC_SCORING_RATIO
                ctc_beam = min(lpz.shape[-1], int(beam * CTC_SCORING_RATIO))
            else:
                ctc_beam = lpz.shape[-1]
        hyps = [hyp]
        ended_hyps = []

        import six
        traced_decoder = None
        for i in six.moves.range(maxlen):
            logging.debug('position ' + str(i))

            hyps_best_kept = []
            for hyp in hyps:
                vy.unsqueeze(1)
                vy[0] = hyp['yseq'][i + start]

                # get nbest local scores and their ids
                ys_mask = subsequent_mask(i + start + 1).unsqueeze(0)
                ys = torch.tensor(hyp['yseq']).unsqueeze(0)
                # FIXME: jit does not match non-jit result
                if use_jit:
                    if traced_decoder is None:
                        traced_decoder = torch.jit.trace(self.decoder.recognize, (ys, ys_mask, enc_output))
                    local_att_scores = traced_decoder(ys, ys_mask, enc_output)
                else:
                    if self.nat:
                        if ret is not None:
                            local_att_scores = ret[:, i]
                        else:
                            ys = torch.tensor(hyp['yseq'][1:] + [self.mask]).unsqueeze(0).long()
                            local_att_scores = self.decoder(ys, ys_mask, enc_output, None)[0] 
                            local_att_scores = torch.log_softmax(local_att_scores[:, -1, torch.arange(self.odim)!=self.mask], dim=-1)
                    else:
                        local_att_scores = self.decoder.recognize(ys, ys_mask, enc_output)

                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp['rnnlm_prev'], vy)
                    local_scores = local_att_scores + recog_args.lm_weight * local_lm_scores
                else:
                    local_scores = local_att_scores

                if lpz is not None:
                    local_best_scores, local_best_ids = torch.topk(
                        local_att_scores, ctc_beam, dim=1)
                    ctc_scores, ctc_states = ctc_prefix_score(
                        hyp['yseq'][start:], local_best_ids[0], hyp['ctc_state_prev'])
                    local_scores = \
                        (1.0 - ctc_weight) * local_att_scores[:, local_best_ids[0]] \
                        + ctc_weight * torch.from_numpy(ctc_scores - hyp['ctc_score_prev'])
                    if rnnlm:
                        local_scores += recog_args.lm_weight * local_lm_scores[:, local_best_ids[0]]
                    local_best_scores, joint_best_ids = torch.topk(local_scores, beam, dim=1)
                    local_best_ids = local_best_ids[:, joint_best_ids[0]]
                else:
                    local_best_scores, local_best_ids = torch.topk(local_scores, beam, dim=1)

                for j in six.moves.range(beam):
                    new_hyp = {}
                    new_hyp['score'] = hyp['score'] + float(local_best_scores[0, j])
                    new_hyp['yseq'] = [0] * (1 + len(hyp['yseq']))
                    new_hyp['yseq'][:len(hyp['yseq'])] = hyp['yseq']
                    new_hyp['yseq'][len(hyp['yseq'])] = int(local_best_ids[0, j])
                    if rnnlm:
                        new_hyp['rnnlm_prev'] = rnnlm_state
                    if lpz is not None:
                        new_hyp['ctc_state_prev'] = ctc_states[joint_best_ids[0, j]]
                        new_hyp['ctc_score_prev'] = ctc_scores[joint_best_ids[0, j]]
                    new_hyp['local_score'] = hyp['local_score'][:]
                    new_hyp['local_score'].append(local_best_scores[0, j].item())
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x['score'], reverse=True)[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug('number of pruned hypothes: ' + str(len(hyps)))
            if char_list is not None:
                logging.debug(
                    'best hypo: ' + ''.join([char_list[int(x)] for x in hyps[0]['yseq'][start:]]))

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info('adding <eos> in the last postion in the loop')
                for hyp in hyps:
                    hyp['yseq'].append(self.eos)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp['yseq'][-1] == self.eos:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp['yseq']) > minlen:
                        hyp['score'] += (i + 1) * penalty
                        if rnnlm:  # Word LM needs to add final <eos> score
                            hyp['score'] += recog_args.lm_weight * rnnlm.final(
                                hyp['rnnlm_prev'])
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)
            if len(remained_hyps) == 0:
                remained_hyps = hyps

            # end detection
            from espnet.nets.e2e_asr_common import end_detect
            if end_detect(ended_hyps, i) and recog_args.maxlenratio == 0.0:
                logging.info('end detected at %d', i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug('remeined hypothes: ' + str(len(hyps)))
            else:
                logging.info('no hypothesis. Finish decoding.')
                break

            if char_list is not None:
                for hyp in hyps:
                    logging.debug(
                        'hypo: ' + ''.join([char_list[int(x)] for x in hyp['yseq'][start:]]))

            logging.debug('number of ended hypothes: ' + str(len(ended_hyps)))

        nbest_hyps = sorted(
            ended_hyps, key=lambda x: x['score'], reverse=True)[:min(len(ended_hyps), recog_args.nbest)]

        # check number of hypotheis
        #if len(nbest_hyps) == 0:
        #    logging.warning('there is no N-best results, perform recognition again with smaller minlenratio.')
        #    # should copy becasuse Namespace will be overwritten globally
        #    recog_args = Namespace(**vars(recog_args))
        #    recog_args.minlenratio = max(0.0, recog_args.minlenratio - 0.1)
        #    return self.recognize(feat, recog_args, char_list, rnnlm)

        logging.info('total log probability: ' + str(nbest_hyps[0]['score']))
        logging.info('normalized log probability: ' + str(nbest_hyps[0]['score'] / len(nbest_hyps[0]['yseq'])))
        return nbest_hyps

    def recognize(self, feat, recog_args, char_list=None, rnnlm=None, use_jit=False):
        """recognize feat.

        :param ndnarray x: input acouctic feature (B, T, D) or (T, D)
        :param namespace recog_args: argment namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list

        TODO(karita): do not recompute previous attention for faster decoding
        """
        if self.nat and recog_args.order != 'left_right':
            xs, ys = feat
            self.eval()

            length = recog_args.maxlength

            xs_pad = torch.from_numpy(xs[0]).unsqueeze(0)
            ilens = torch.from_numpy(np.array([xs[0].shape[0]]))
            ys_in_pad = torch.zeros(1, length, dtype=torch.long) + self.mask
            ys_in_pad[:, -1] = self.mask
            xs_mask = (~make_pad_mask(ilens.tolist())).to(xs_pad.device).unsqueeze(-2)
            hs_pad, hs_mask = self.encoder(xs_pad, xs_mask)
            div = length // (recog_args.T)
            
            pred_pad, pred_mask = self.decoder(ys_in_pad, (ys_in_pad != self.ignore_id).unsqueeze(-2), hs_pad, hs_mask)

            if recog_args.truncate:
                pos_eos = torch.nonzero(pred_pad[0].argmax(dim=-1) == self.eos)
                if torch.sum(pos_eos) > 0 and recog_args.T > 1:
                    length = max(1, min(length, pos_eos[0].item() + 3))
                    ys_in_pad = ys_in_pad[:, :length]
                    pred_pad = pred_pad[:, :length]

            ret = torch.zeros(1, length, self.odim - 1)
            ret[0, :, :] = torch.log_softmax(pred_pad[0, :, torch.arange(self.odim)!=self.mask], dim=-1)
            logging.info('step {0} {1}: '.format(0, ''.join([char_list[int(x)] for x in ret[0].argmax(dim=-1).numpy().tolist()]).replace('<space>', ' ')))
            for t in range(1, recog_args.T):
                #hyps = self.beamsearch(hs_pad, recog_args, ret, char_list, rnnlm, use_jit)
                #p = torch.tensor(hyps[0]['local_score']).unsqueeze(0)
                #ass = torch.tensor(hyps[0]['yseq'][1:]).unsqueeze(0)
                
                #mask predict
                if recog_args.order == 'mask_predict':
                    p, ass = ret.max(dim=-1)
                    _, idx = torch.topk(p[0], p.size(-1) * (recog_args.T - t) // recog_args.T, largest=False, dim=-1)

                    if ass.size(1) <= ys_in_pad.size(1):
                        ys_in_pad[:, :ass.size(1)] = ass
                    else:
                        ys_in_pad = ass
                    ys_in_pad[0, idx] = self.mask
                    logging.info('step {0} {1} input: '.format(t, ''.join([char_list[int(x)] if int(x) != self.mask else '<mask>' for x in ys_in_pad[0].numpy().tolist()]).replace('<space>', ' ')))
                    pred_pad, pred_mask = self.decoder(ys_in_pad, (ys_in_pad != self.ignore_id).unsqueeze(-2), hs_pad, hs_mask)
                    
                    ret[0, idx] = torch.log_softmax(pred_pad[0, idx][:, torch.arange(self.odim)!=self.mask], dim=-1)
                    logging.info('step {0} {1} result: '.format(t, ''.join([char_list[int(x)] for x in ret[0].argmax(dim=-1).numpy().tolist()]).replace('<space>', ' ')))

                #easy first
                if recog_args.order == 'easy_first':
                    p, ass = pred_pad.max(dim=-1)
                    mask_idx = torch.nonzero(ys_in_pad[0] == self.mask).flatten()
                    probs, idx = torch.topk(p[0, mask_idx], min((p.size(-1) + recog_args.T - 1) // recog_args.T, mask_idx.size(0)), largest=True, dim=-1, sorted=False)
                    ys_in_pad[0, mask_idx[idx]] = ass[0, mask_idx[idx]]
                    ret[0, mask_idx[idx]] = torch.log_softmax(pred_pad[0, mask_idx[idx]][:, torch.arange(self.odim)!=self.mask], dim=-1)
                    pred_pad, pred_mask = self.decoder(ys_in_pad, (ys_in_pad != self.ignore_id).unsqueeze(-2), hs_pad, hs_mask)
                    logging.info('step {0}: {1}'.format(t, ''.join([char_list[int(x)] if int(x) != self.mask else '__' for x in ys_in_pad[0].numpy().tolist()]).replace('<space>', ' ')))
                    logging.info('result {0}: {1}'.format(t, ''.join([char_list[int(x)] for x in ret[0].argmax(dim=-1).numpy().tolist()]).replace('<space>', ' ')))

            return self.beamsearch(hs_pad, recog_args, ret, char_list, rnnlm, use_jit)
        else:
            if self.nat:
                feat = feat[0][0]
            enc_output = self.encode(feat).unsqueeze(0)
            return self.beamsearch(enc_output, recog_args, None, char_list, rnnlm, use_jit)

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad, ys_pad_asr=None):
        """E2E attention calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded character id sequence tensor (B, Lmax)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).
        :rtype: float ndarray
        """
        with torch.no_grad():
            self.forward(xs_pad, ilens, ys_pad)
        ret = dict()
        for name, m in self.named_modules():
            if isinstance(m, MultiHeadedAttention):
                ret[name] = m.attn.cpu().numpy()
        return ret
