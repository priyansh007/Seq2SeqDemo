import supportFiles.Dataset as Dataset
import supportFiles.HelperFunctions as HelperFunctions
import supportFiles.Seq2Seq as Seq2Seq
import torch
import torch.nn as nn
from tqdm import tqdm


def startTraining():
    input_texts, target_texts = HelperFunctions.read_dataset("Dataset/eng-chin.txt")
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

    dataset = Dataset.Dataset(
        X = english,
        Y = chinese,
        in_tknz = input_tokenizer, out_tknz = output_tokenizer,
        in_maxlen = in_maxlen, out_maxlen = out_maxlen
    )

    dataloader = HelperFunctions.get_dataloader(dataset, batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2Seq.Seq2Seq(
        in_maxlen = in_maxlen,
        out_maxlen = out_maxlen,
        n_hidden = n_hidden,
        enc_n_class = enc_n_class,
        dec_n_class = dec_n_class,
        d_model = d_model,
        num_layers = 1,
    )
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)

    epochs = 200

    torch.cuda.empty_cache()
    model.train()
    model.to(device)
    loss_records = []
    for epoch in tqdm(range(epochs)):
        # Runs the model and calculates loss
        loss = 0
        for _, (enc_in, dec_in, dec_out) in enumerate(dataloader):
            enc_h_0 = model.init_enc_hidden_GRU(batch_size, device)
            enc_in, dec_in = enc_in.to(device), dec_in.to(device)
            dec_out = dec_out.to(device)

            pred = model(enc_in, enc_h_0, dec_in)
            for i in range(len(dec_out)):
                loss += criterion(pred[i], dec_out[i])
        if (epoch) % 10 == 0:
            print(f"Epoch: {epoch}, Loss: {loss}")
        if (epoch) % 5 == 0:
            loss_records.append(loss)
        optim.zero_grad()
        loss.backward()
        optim.step()
        torch.save(model.state_dict(), "./myLib/seq2seq.pt")

startTraining()