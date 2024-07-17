import torch

debug = True
#valid_image_path = "C:/Users/asus/Desktop/InteligentniSustaviCLIP/data/medical/valid"
#image_path = "C:/Users/asus/Desktop/InteligentniSustaviCLIP/data/medical/train"
image_path = "C:/Users/asus/Desktop/InteligentniSustaviCLIP/data/medical/test"
captions_path = "C:/Users/asus/Desktop/InteligentniSustaviCLIP/data/medical"
batch_size = 32
num_workers = 4
lr = 1e-4
weight_decay = 1e-3
patience = 1
factor = 0.8
epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
early_stopping_patience = 3

model_name = 'resnet50'
image_embedding = 2048
text_encoder_model = "distilbert-base-uncased"
text_embedding = 768
text_tokenizer = "distilbert-base-uncased"
max_length = 200

pretrained = True # for both image encoder and text encoder
trainable = True # for both image encoder and text encoder
temperature = 1.0

# image size
size = 224

# for projection head; used for both image and text encoders
num_projection_layers = 1
projection_dim = 256 
dropout = 0.1