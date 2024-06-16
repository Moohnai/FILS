import torch
import clip
from PIL import Image
import pandas as pd
import json
import os
import open_clip
from tqdm import tqdm

# def text_encoding(text):
#     """
#     text: list of strings
#     """
#     text = clip.tokenize(text[:77]).to(device)
#     with torch.no_grad():
#         text_features = model.encode_text(text)
#     return text_features

# model, _, preprocess = open_clip.create_model_and_transforms('ViT-G-14', pretrained='laion2B-s34B-b88K')
model, _, preprocess = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2b_s39b_b160k')
model = model.to("cuda")


def text_encoding_openclip(text):
    """
    text: list of strings
    """
    # tokenizer = open_clip.get_tokenizer('ViT-bigG-14')
    tokenizer = open_clip.get_tokenizer('ViT-bigG-14')
    text = tokenizer(text[:77]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
    return text_features


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model, _ = clip.load("ViT-L/14", device=device)
    ######epic_kitchens
    # read csv file
    csv_train_path = "/home/mona/FRIL/avion/datasets/EK100/epic-kitchens-100-annotations/EPIC_100_train.csv"
    # csv_val_path = "../../home/mona/SSVLI/dataset/epic_kitchens/EPIC_100_validation.csv"


     
    #noun
    #read csv file pandas
    # df_train = pd.read_csv(csv_train_path)
    # df_val = pd.read_csv(csv_val_path)

    ## preprocess text
    # encoded_text_train_noun = [text_encoding(text) for text in df_train['noun']]
    # encoded_text_val_noun = [text_encoding(text) for text in df_val['noun']]

    #verb
    # encoded_text_train_verbs = [text_encoding(text) for text in df_train['verb']]
    # encoded_text_val_verbs = [text_encoding(text) for text in df_val['verb']]

    # save encoded text in pt file
    # torch.save(encoded_text_train_noun, "../../home/mona/SSVLI/dataset/epic_kitchens/EPIC_100_train_noun_text.pt")
    # torch.save(encoded_text_val_noun, "../../home/mona/SSVLI/dataset/epic_kitchens/EPIC_100_val_noun_text.pt")
    
    # torch.save(encoded_text_train_verbs, "../../home/mona/SSVLI/dataset/epic_kitchens/EPIC_100_train_verb_text.pt")
    # torch.save(encoded_text_val_verbs, "../../home/mona/SSVLI/dataset/epic_kitchens/EPIC_100_val_verb_text.pt")
    
    #action
    csv_action_path_train = "../../../../home/mona/SSVLI/dataset/epic_kitchens/annotation/action/train.csv"
    # csv_action_path_val = "../../home/mona/SSVLI/dataset/epic_kitchens/annotation/action/val.csv"
    generated_csv_path = "/home/mona/FRIL/avion/datasets/EK100/epic_captions_train.csv"
    df_generated = pd.read_csv(generated_csv_path, delimiter=',')
    df_generated['video'] = df_generated['video'].apply(lambda x: x.split('/')[-1].split('.')[0].split('_')[-1])


    df_train = pd.read_csv(csv_action_path_train, header=None, delimiter=' ')
    df_train[0] = df_train[0].apply(lambda x: x.split('/')[-1].split('.')[0].split('_')[-1])
    df_train[1] = df_train[1].apply(lambda x: x.replace("_", " "))


    # df_val = pd.read_csv(csv_action_path_val, header=None, delimiter=' ')

    #read only the second column from df_train
    #replace the "_" with " " in the text
    
    # encoded_label_train_dict_epic = {id:text_encoding_clip(text) for _, (id, text,_) in df_train.iterrows()}

    # encoded_text_train_action = [text_encoding(text) for text in df_train[1].str.replace("_", " ")]
    # encoded_text_val_action = [text_encoding(text) for text in df_val[1].str.replace("_", " ")]

    # torch.save(encoded_text_val_action, "../../home/mona/SSVLI/dataset/epic_kitchens/EPIC_100_val_action_text.pt")
    encoded_video_cation_dict={}
    encoded_image_cation_dict={}
    encoded_mixed_cation_dict={}
    for id in tqdm(df_train[0], total=len(df_train[0])):
        row = df_generated.loc[df_generated['video'] == id]
        encoded_video_cation_dict[id] = text_encoding_openclip(row['video_caption'].values[0])
        encoded_image_cation_dict[id] = text_encoding_openclip(row['image_caption'].values[0])
        encoded_mixed_cation_dict[id] = text_encoding_openclip(row['mixed_caption'].values[0])
        
    # torch.save(encoded_label_train_dict_epic, "/home/mona/FRIL/avion/datasets/EK100/O_epic_train_label_text_dict.pt")
    torch.save(encoded_video_cation_dict, "/home/mona/FRIL/avion/datasets/EK100/O_epic_train_video_caption_text_dict.pt")
    torch.save(encoded_image_cation_dict, "/home/mona/FRIL/avion/datasets/EK100/O_epic_train_image_caption_text_dict.pt")
    torch.save(encoded_mixed_cation_dict, "/home/mona/FRIL/avion/datasets/EK100/O_epic_train_mixed_caption_text_dict.pt")









# ######################################## SSV2
# #read a json file and create a csv file with id and label
# # json_path = "../../home/mona/SSVLI/dataset/ssv2/train.json"
# # csv_path = "../../home/mona/SSVLI/dataset/ssv2"
# # #read json file
# # f = open(json_path, 'r')
# # train_label = json.load(f)


# # train_lable_df = {'id':[], 'label_name':[]}
# # for i in train_label:
# #     id = i['id']    
# #     label_name = i['label']
# #     train_lable_df['id'].append(id)
# #     train_lable_df['label_name'].append(label_name)
# # train_label_df = pd.DataFrame(train_lable_df)
# # train_label_df.to_csv(path_or_buf=os.path.join(csv_path, "labels_train.csv"), sep=' ', na_rep='', float_format=None, 
# # columns=None, header=False, index=False, index_label=None, mode='w', encoding=None, 
# # compression='infer', quoting=None, quotechar='"', 
# # chunksize=None, date_format=None, doublequote=True, escapechar=None, 
# # decimal='.', errors='strict', storage_options=None)


# if __name__ == "__main__":
    
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model, _ = clip.load("ViT-L/14", device=device)

#     #read csv file
#     csv_train_path = "../../home/mona/SSVLI/dataset/ssv2/labels_train.csv"
#     generated_csv_path = "../../home/mona/SSVLI/dataset/ssv2/ssv2_captions.csv"
#     df_label = pd.read_csv(csv_train_path, delimiter=' ', header=None)
#     df_generated = pd.read_csv(generated_csv_path, delimiter=',')
#     df_generated['video'] = df_generated['video'].apply(lambda x: x.split('/')[-1].split('.')[0])


#     encoded_label_train_dict_ssv2 = {id:text_encoding(text) for _, (id, text) in df_label.iterrows()}

#     encoded_video_cation_dict={}
#     encoded_image_cation_dict={}
#     encoded_mixed_cation_dict={}
#     for id in df_label[0]:
#         if str(id) in df_generated['video'].values:
#             row = df_generated.loc[df_generated['video'] == str(id)]
#             encoded_video_cation_dict[id] = text_encoding(row['video_caption'].values[0])
#             encoded_image_cation_dict[id] = text_encoding(row['image_caption'].values[0])
#             encoded_mixed_cation_dict[id] = text_encoding(row['mixed_caption'].values[0])

#     # torch.save(encoded_label_train_dict_ssv2, "../../home/mona/SSVLI/dataset/ssv2/ssv2_train_label_text_dict.pt")
#     torch.save(encoded_video_cation_dict, "../../home/mona/SSVLI/dataset/ssv2/ssv2_train_video_caption_text_dict.pt")
#     torch.save(encoded_image_cation_dict, "../../home/mona/SSVLI/dataset/ssv2/ssv2_train_image_caption_text_dict.pt")
#     torch.save(encoded_mixed_cation_dict, "../../home/mona/SSVLI/dataset/ssv2/ssv2_train_mixed_caption_text_dict.pt")

