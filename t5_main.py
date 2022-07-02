import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from transformers.optimization import Adafactor
import warnings

from IPython.display import HTML, display

warnings.filterwarnings('ignore')


def progress(loss, value, max=100):
    return HTML(""" Batch loss :{loss}
        <progress
            value='{value}'
            max='{max}',
            style='width: 100%'
        >
            {value}
        </progress>
    """.format(loss=loss, value=value, max=max))


if __name__ == "__main__":

    train_df = pd.read_csv('/Users/danilamarius-cristian/PycharmProjects/pythonProject3/train_berteilung.csv', index_col=[0])

    batch_size = 8
    num_of_batches = int(len(train_df) / batch_size)
    num_of_epochs = 4

    if torch.cuda.is_available():
        dev = torch.device("cuda:0")
        print("Running on the GPU")
    else:
        dev = torch.device("cpu")
        print("Running on the CPU")

    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    model = T5ForConditionalGeneration.from_pretrained('t5-base', return_dict=True)

    model.to(dev)

    optimizer = Adafactor(
        model.parameters(),
        lr=1e-3,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        decay_rate=-0.8,
        beta1=None,
        weight_decay=0.0,
        relative_step=False,
        scale_parameter=False,
        warmup_init=False
    )

    model.train()

    loss_per_10_steps = []
    for epoch in range(1, num_of_epochs + 1):
        print('Running epoch: {}'.format(epoch))

        running_loss = 0

        out = display(progress(1, num_of_batches + 1), display_id=True)
        for i in range(num_of_batches):
            inputbatch = []
            labelbatch = []
            new_df = train_df[i * batch_size:i * batch_size + batch_size]
            for indx, row in new_df.iterrows():
                input = 'WebNLG: ' + row['input_text'] + '</s>'
                labels = row['target_text'] + '</s>'
                inputbatch.append(input)
                labelbatch.append(labels)
            inputbatch = tokenizer.batch_encode_plus(inputbatch, padding=True, max_length=400, return_tensors='pt')[
                "input_ids"]
            labelbatch = tokenizer.batch_encode_plus(labelbatch, padding=True, max_length=400, return_tensors="pt")[
                "input_ids"]
            inputbatch = inputbatch.to(dev)
            labelbatch = labelbatch.to(dev)

            # clear out the gradients of all Variables
            optimizer.zero_grad()

            # Forward propogation
            outputs = model(input_ids=inputbatch, labels=labelbatch)
            loss = outputs.loss
            loss_num = loss.item()
            logits = outputs.logits
            running_loss += loss_num
            if i % 10 == 0:
                loss_per_10_steps.append(loss_num)
            # out.update(progress(loss_num, i, num_of_batches + 1))

            print(" loss_num: {}; i: {}; batches {} ".format(loss_num, i, num_of_batches + 1))

            # calculating the gradients
            loss.backward()

            # updating the params
            optimizer.step()

        running_loss = running_loss / int(num_of_batches)
        print('Epoch: {} , Running loss: {}'.format(epoch, running_loss))

    print("===== Saving the model =====")
    torch.save(model.state_dict(), 'pytoch_model_berteilung.bin')
