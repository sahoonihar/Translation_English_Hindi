import torch
import copy
seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
    


def beam_search(sentence, model, src_field, src_tokenizer, trg_field, trg_vcb_sz, k, max_ts=50, device="cpu"):
    # Tokenize the input sentence
    sentence_tok = src_tokenizer(sentence)

    # Add <sos> and <eos> in beginning and end respectively
    sentence_tok.insert(0, src_field.init_token)
    sentence_tok.append(src_field.eos_token)

    # Converting text to indices
    src_tok = torch.tensor([src_field.vocab.stoi[token] for token in sentence_tok], dtype=torch.long).unsqueeze(0).to(device)
    trg_tok = torch.tensor([trg_field.vocab.stoi[trg_field.init_token]], dtype=torch.long).unsqueeze(0).to(device)

    # Setting 'eos' flag for target sentence
    eos = trg_field.vocab.stoi[trg_field.eos_token]

    # Store for top 'k' translations
    trans_store = {}

    store_seq_id = None
    store_seq_prob = None
    for ts in range(max_ts):
        if ts == 0:
            with torch.no_grad():
                out = model(src_tok, trg_tok)  # [1, trg_vcb_sz]
            topk = torch.topk(torch.log(torch.softmax(out, dim=-1)), dim=-1, k=k)
            seq_id = torch.empty(size=(k, ts + 2), dtype=torch.long)
            seq_id[:, :ts + 1] = trg_tok
            seq_id[:, ts + 1] = topk.indices
            seq_prob = topk.values
            if eos in seq_id[:, ts + 1]:
                trans_store[seq_prob[:, seq_id[:, ts + 1] == eos].squeeze()] = seq_id[seq_id[:, ts + 1] == eos, :].squeeze()
                store_seq_id = copy.deepcopy(seq_id[seq_id[:, ts + 1] != eos, :]).to(device)
                store_seq_prob = copy.deepcopy(seq_prob[:, seq_id[:, ts + 1] != eos].squeeze()).to(device)
            else:
                store_seq_id = copy.deepcopy(seq_id).to(device)
                store_seq_prob = copy.deepcopy(seq_prob).to(device)
        else:
            src_tok = src_tok.squeeze()
            src = src_tok.expand(size=(store_seq_id.shape[-2], len(src_tok))).to(device)
            with torch.no_grad():
                out = model(src, store_seq_id)
            out = torch.log(torch.softmax(out[:, -1, :], dim=-1))  # [k, trg_vcb_sz]
            all_comb = (store_seq_prob.view(-1, 1) + out).view(-1)
            all_comb_idx = torch.tensor([(x, y) for x in range(store_seq_id.shape[-2]) for y in range(trg_vcb_sz)])
            topk = torch.topk(all_comb, dim=-1, k=k)
            top_seq_id = all_comb_idx[topk.indices.squeeze()]
            top_seq_prob = topk.values
            seq_id = torch.empty(size=(k, ts + 2), dtype=torch.long)
            seq_id[:, :ts + 1] = torch.tensor([store_seq_id[i.tolist()].tolist() for i, y in top_seq_id])
            seq_id[:, ts + 1] = torch.tensor([y.tolist() for i, y in top_seq_id])
            seq_prob = top_seq_prob
            if eos in seq_id[:, ts + 1]:
                trans_store[seq_prob[seq_id[:, ts + 1] == eos].squeeze()] = seq_id[seq_id[:, ts + 1] == eos, :].squeeze()
                store_seq_id = copy.deepcopy(seq_id[seq_id[:, ts + 1] != eos, :]).to(device)
                store_seq_prob = copy.deepcopy(seq_prob[seq_id[:, ts + 1] != eos].squeeze()).to(device)
            else:
                store_seq_id = copy.deepcopy(seq_id).to(device)
                store_seq_prob = copy.deepcopy(seq_prob).to(device)
        if len(trans_store) == k:
            break

    if len(trans_store) == 0:
        best_translation = store_seq_id[0]
    else:
        best_translation = trans_store[max(trans_store)]
    return " ".join([trg_field.vocab.itos[w] for w in best_translation[1:-1]])
    

def beam_search_bpe(sentence, model, src_field, src_tokenizer, trg_field, trg_vcb_sz, k, max_ts=50, device="cpu"):
    # Tokenize the input sentence
    sentence_tok = src_tokenizer(sentence)

    # Add <sos> and <eos> in beginning and end respectively
    sentence_tok.insert(0, src_field.init_token)
    sentence_tok.append(src_field.eos_token)

    # Converting text to indices
    src_tok = torch.tensor([src_field.vocab.stoi[token] for token in sentence_tok], dtype=torch.long).unsqueeze(0).to(device)
    trg_tok = torch.tensor([trg_field.vocab.stoi[trg_field.init_token]], dtype=torch.long).unsqueeze(0).to(device)

    # Setting 'eos' flag for target sentence
    eos = trg_field.vocab.stoi[trg_field.eos_token]

    # Store for top 'k' translations
    trans_store = {}

    store_seq_id = None
    store_seq_prob = None
    for ts in range(max_ts):
        if ts == 0:
            with torch.no_grad():
                out = model(src_tok, trg_tok)  # [1, trg_vcb_sz]
            topk = torch.topk(torch.log(torch.softmax(out, dim=-1)), dim=-1, k=k)
            seq_id = torch.empty(size=(k, ts + 2), dtype=torch.long)
            seq_id[:, :ts + 1] = trg_tok
            seq_id[:, ts + 1] = topk.indices
            seq_prob = topk.values
            if eos in seq_id[:, ts + 1]:
                trans_store[seq_prob[:, seq_id[:, ts + 1] == eos].squeeze()] = seq_id[seq_id[:, ts + 1] == eos, :].squeeze()
                store_seq_id = copy.deepcopy(seq_id[seq_id[:, ts + 1] != eos, :]).to(device)
                store_seq_prob = copy.deepcopy(seq_prob[:, seq_id[:, ts + 1] != eos].squeeze()).to(device)
            else:
                store_seq_id = copy.deepcopy(seq_id).to(device)
                store_seq_prob = copy.deepcopy(seq_prob).to(device)
        else:
            src_tok = src_tok.squeeze()
            src = src_tok.expand(size=(store_seq_id.shape[-2], len(src_tok))).to(device)
            with torch.no_grad():
                out = model(src, store_seq_id)
            out = torch.log(torch.softmax(out[:, -1, :], dim=-1))  # [k, trg_vcb_sz]
            all_comb = (store_seq_prob.view(-1, 1) + out).view(-1)
            all_comb_idx = torch.tensor([(x, y) for x in range(store_seq_id.shape[-2]) for y in range(trg_vcb_sz)])
            topk = torch.topk(all_comb, dim=-1, k=k)
            top_seq_id = all_comb_idx[topk.indices.squeeze()]
            top_seq_prob = topk.values
            seq_id = torch.empty(size=(k, ts + 2), dtype=torch.long)
            seq_id[:, :ts + 1] = torch.tensor([store_seq_id[i.tolist()].tolist() for i, y in top_seq_id])
            seq_id[:, ts + 1] = torch.tensor([y.tolist() for i, y in top_seq_id])
            seq_prob = top_seq_prob
            if eos in seq_id[:, ts + 1]:
                trans_store[seq_prob[seq_id[:, ts + 1] == eos].squeeze()] = seq_id[seq_id[:, ts + 1] == eos, :].squeeze()
                store_seq_id = copy.deepcopy(seq_id[seq_id[:, ts + 1] != eos, :]).to(device)
                store_seq_prob = copy.deepcopy(seq_prob[seq_id[:, ts + 1] != eos].squeeze()).to(device)
            else:
                store_seq_id = copy.deepcopy(seq_id).to(device)
                store_seq_prob = copy.deepcopy(seq_prob).to(device)
        if len(trans_store) == k:
            break

    if len(trans_store) == 0:
        best_translation = store_seq_id[0]
    else:
        best_translation = trans_store[max(trans_store)]
        
    word_translation = [trg_field.vocab.itos[w] for w in best_translation[1:-1]]
    subword_translation = " ".join([trg_field.vocab.itos[w] for w in best_translation[1:-1]])
    return word_translation, subword_translation
    #return " ".join([trg_field.vocab.itos[w] for w in best_translation[1:-1]])
