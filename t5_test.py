import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

if torch.cuda.is_available():
    dev = torch.device("cuda:0")
    print("Running on the GPU")
else:
    dev = torch.device("cpu")
    print("Running on the CPU")


def generate_for_berteilung(context_words):
    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    model = T5ForConditionalGeneration.from_pretrained('/Users/danilamarius-cristian/PycharmProjects/pythonProject3/pytoch_model_berteilung.bin', return_dict=True,
                                                       config='/Users/danilamarius-cristian/PycharmProjects/pythonProject3/t5-base-config.json')


    model.eval()
    input_ids = tokenizer.encode(" | ".join(context_words[:-1]) + ' ' + context_words[-1], return_tensors="pt")
    input_ids=input_ids.to(dev)
    outputs = model.generate(input_ids)
    return tokenizer.decode(outputs[0])


def generate_for_anamnesis(context_words):
    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    model = T5ForConditionalGeneration.from_pretrained(
        '/Users/danilamarius-cristian/PycharmProjects/pythonProject3/pytoch_model.bin', return_dict=True,
        config='/Users/danilamarius-cristian/PycharmProjects/pythonProject3/t5-base-config.json')

    model.eval()
    input_ids = tokenizer.encode(" | ".join(context_words[:-1]) + ' ' + context_words[-1],
                                 return_tensors="pt")  # Batch size 1
    input_ids = input_ids.to(dev)
    outputs = model.generate(input_ids)
    return tokenizer.decode(outputs[0])


if __name__ == "__main__":



    tokenizer = T5Tokenizer.from_pretrained('t5-base')

    model = T5ForConditionalGeneration.from_pretrained(
        '/Users/danilamarius-cristian/PycharmProjects/pythonProject3/pytoch_model.bin', return_dict=True,
        config='/Users/danilamarius-cristian/PycharmProjects/pythonProject3/t5-base-config.json')

    model.eval()
    input_ids = tokenizer.encode("present|hours|consultation|family|patient", return_tensors="pt")  # Batch size 1
    input_ids=input_ids.to(dev)
    outputs = model.generate(input_ids)
    print(tokenizer.decode(outputs[0]))

