from nltk.tokenize import WordPunctTokenizer
import torch
import HelperFunctions
import Seq2Seq

def translate(eng_sent):
    # Set up the inputs and variables
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tk = WordPunctTokenizer()
    input_texts, target_texts = HelperFunctions.read_dataset("./eng-chin.txt")
    english,chinese = HelperFunctions.tokenize_sentences(input_texts, target_texts)
    input_tokenizer, output_tokenizer = HelperFunctions.build_vocab(english,chinese)
    max_english_length = HelperFunctions.count_max_sentence_len(english)
    max_chinese_length = HelperFunctions.count_max_sentence_len(chinese)
    in_maxlen = max_english_length + 1
    out_maxlen = max_chinese_length + 1
    n_hidden = 32 # Number of "neurons" per layer
    d_model = 64 # Number of embedding dimensions to represent each word
    enc_n_class = len(input_tokenizer.dict) # OR... vocab size of englisth -> 199
    dec_n_class = len(output_tokenizer.dict) # OR... vocab size of chinese -> 317
    batch_size = 1

    model = Seq2Seq.Seq2Seq(
        in_maxlen = in_maxlen,
        out_maxlen = out_maxlen,
        n_hidden = n_hidden,
        enc_n_class = enc_n_class,
        dec_n_class = dec_n_class,
        d_model = d_model,
        num_layers = 1,
    )
    model_path = "./myLib/seq2seq.pt"
    model_state_dict = torch.load(model_path)
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()
    
    eng_sent = tk.tokenize(eng_sent.lower()) + ["<EOS>"]
    eng_sent = input_tokenizer.transform(eng_sent, max_len=in_maxlen, pad_first=False)
    dec_in = (["<SOS>"] + ["<PAD>"]*out_maxlen)[:out_maxlen]
    dec_in = output_tokenizer.transform(dec_in, max_len=out_maxlen, pad_first=False)

    enc_h_0 = model.init_enc_hidden_GRU(batch_size, device)
    eng_sent, dec_in = torch.LongTensor(eng_sent), torch.LongTensor(dec_in)
    eng_sent = eng_sent.unsqueeze(0)
    dec_in = dec_in.unsqueeze(0)
    eng_sent, dec_in = eng_sent.to(device), dec_in.to(device)
    # Run the model
    with torch.no_grad():
        # eng_sent: [1(b), 26(in_maxlen)]
        embedded_X = model.embed_enc(eng_sent)
        # embedded_X: [26(in_maxlen), 1(b), 64(d_model)] <- [1(b), 26(in_maxlen), 64(d_model)]
        embedded_X = embedded_X.permute(1, 0, 2)
        _, memory = model.encoder(embedded_X, enc_h_0)
        pred_loc = 0
        for i in range(out_maxlen-1):
            embedded_Y = model.embed_dec(dec_in)
            embedded_Y = embedded_Y.permute(1, 0, 2)
            outputs, _ = model.decoder(embedded_Y, memory)
            outputs = outputs.permute(1, 0, 2)
            pred = model.fc(outputs)
            pred = pred[0][pred_loc].topk(1)[1].item()
            pred_loc += 1
            if pred == 2:
                dec_in[0][pred_loc] = pred
                break
            else:
                dec_in[0][pred_loc] = pred

    return output_tokenizer.inverse_transform(dec_in[0], is_tensor=True)


sent = "I dont know"
translated = translate(sent)
translated_sent = "".join([word for word in translated if word != "<SOS>" and word != "<EOS>"and word != "<PAD>" and word!="<UNK>"])
print(f"{sent} -> \n{translated_sent}")