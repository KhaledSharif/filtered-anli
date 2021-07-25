from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
import pandas as pd
import time
import numpy as np

if __name__ == '__main__':

    print("using gpu?", torch.cuda.is_available())

    start = time.time()
    max_length = 256
    # with open('./data/anli_v1.0/R1/test.jsonl', 'r') as r1_train_jsonl:
    #     r1_train = [json.loads(line) for line in r1_train_jsonl]
    r1_train = pd.read_json(path_or_buf='./data/anli_v1.0/R1/train.jsonl', lines=True)
    r2_train = pd.read_json(path_or_buf='./data/anli_v1.0/R2/train.jsonl', lines=True)
    r3_train = pd.read_json(path_or_buf='./data/anli_v1.0/R3/train.jsonl', lines=True)

    r1_train_context = r1_train["context"]
    r2_train_context = r2_train["context"]
    r3_train_context = r3_train["context"]

    r1_train_hypothesis = r1_train["hypothesis"]
    r2_train_hypothesis = r2_train["hypothesis"]
    r3_train_hypothesis = r3_train["hypothesis"]

    # print(r1_train_hypothesis[0])
    #
    premise = "Two women are embracing while holding to go packages."
    hypothesis = "The men are fighting outside a deli."
    # print(premise)

    hg_model_hub_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
    # hg_model_hub_name = "ynie/albert-xxlarge-v2-snli_mnli_fever_anli_R1_R2_R3-nli"
    # hg_model_hub_name = "ynie/bart-large-snli_mnli_fever_anli_R1_R2_R3-nli"
    # hg_model_hub_name = "ynie/electra-large-discriminator-snli_mnli_fever_anli_R1_R2_R3-nli"
    # hg_model_hub_name = "ynie/xlnet-large-cased-snli_mnli_fever_anli_R1_R2_R3-nli"

    tokenizer = AutoTokenizer.from_pretrained(hg_model_hub_name)
    model = AutoModelForSequenceClassification.from_pretrained(hg_model_hub_name)

    embeddings = []
#
    m = int(len(r1_train.index)/100) #169.48 -> 169
    print(m)
    for i in np.arange(m+1): # 0 ~ 169 inclusive
        embedding = []
        print(min(i * 100 + 100, len(r1_train.index)))
        for j in np.arange(i*100, min(i*100 + 100, len(r1_train.index))):
            tokenized_input_seq_pair = tokenizer.encode_plus(r2_train_context[j], r2_train_hypothesis[j],
                                                         max_length=max_length,
                                                         return_token_type_ids=True, truncation=True)
            input_ids = torch.Tensor(tokenized_input_seq_pair['input_ids']).long().unsqueeze(0)
            token_type_ids = torch.Tensor(tokenized_input_seq_pair['token_type_ids']).long().unsqueeze(0)
            attention_mask = torch.Tensor(tokenized_input_seq_pair['attention_mask']).long().unsqueeze(0)

            outputs = model(input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=None)
            # print(outputs[0][0].detach().numpy())
            embedding.append(outputs[0][0].detach().numpy())
        print(len(embedding))
        "----------------------end of batch---------------------"
        np.save("./embedding_files/R2/batch-"+str(i), embedding)

        print(embedding)

    m = 169
    for i in np.arange(m-3, m+1):
        x = np.load("./embedding_files/R2/batch-"+str(i)+".npy")
        print(len(x))
        print(x)
        # np.savez("file"+str(i), temp)


    # print(embeddings)
    # print(len(embeddings))
    # print(embeddings.shape)
    # embeddings = [item for embedding in embeddings for item in embedding]
    # print(embeddings)
    # print(len(embeddings))
    # end = time.time()
    print("time:", end - start)

