import torch
import torch.nn as nn
import gradio as gr
from transformers import AutoTokenizer, AutoModel

class TransformerModel(nn.Module):
    def __init__(self, modelName='roberta-base', numClasses=5):
        super(TransformerModel, self).__init__()
        self.bert = AutoModel.from_pretrained(modelName)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(768, numClasses)
        
    def forward(self, ids, mask):
        out = self.bert(ids, attention_mask=mask)
        return self.fc(self.drop(out.pooler_output))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained('./roberta_emotion_tokenizer')

model = TransformerModel(modelName='roberta-base', numClasses=5)
model.load_state_dict(torch.load('roberta_emotion_model.pth', map_location=device))
model.to(device)
model.eval()

labels = ['anger', 'fear', 'joy', 'sadness', 'surprise']

def predict_emotion(text):
    if not text:
        return "Please enter text."
    
    enc = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=64,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    ids = enc['input_ids'].flatten().unsqueeze(0).to(device)
    mask = enc['attention_mask'].flatten().unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(ids, mask)
        probs = torch.sigmoid(output).squeeze().cpu().numpy()
    
    return {label: float(prob) for label, prob in zip(labels, probs)}

app = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Textbox(lines=2, placeholder="Type a sentence here..."),
    outputs=gr.Label(num_top_classes=5),
    title="Emotion Classification",
)

app.launch()