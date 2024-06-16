import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt


# Euclidean distance 
def euclidean_distance(x,y):
    return torch.dist(x,y)

# Cosine distance
def cosine_distance(x,y):
    return torch.nn.functional.cosine_similarity(x,y)

# Dot product
def dot_product(x,y):
    x = x.reshape(-1)
    y = y.reshape(-1)
    return torch.dot(x,y)

# train_lable_embed_path_epic = '../../home/mona/SSVLI/dataset/epic_kitchens/epic_train_label_text_dict.pt'

train_org_label_embed_path_epic = '/home/mona/FRIL/FILS/datasets/EK100/epic_train_label_text_dict.pt'
# epic_train_image_caption_path = '../../home/mona/FRIL/FILS/datasets/EK100/epic_train_image_caption_text_dict.pt'
# epic_train_video_caption_path = '../../home/mona/FRIL/FILS/datasets/EK100/epic_train_video_caption_text_dict.pt'
# epic_train_mixed_caption_path = '../../home/mona/FRIL/FILS/datasets/EK100/epic_train_mixed_caption_text_dict.pt'
epic_train_image_caption_path = '/home/mona/FRIL/FILS/datasets/EK100/vifi_epic_train_image_caption_text_dict.pt'
epic_train_video_caption_path = '/home/mona/FRIL/FILS/datasets/EK100/vifi_epic_train_video_caption_text_dict.pt'
epic_train_mixed_caption_path = '/home/mona/FRIL/FILS/datasets/EK100/vifi_epic_train_mixed_caption_text_dict.pt'


epic_train_lable_embed = torch.load(train_org_label_embed_path_epic)
epic_train_image_caption_embed = torch.load(epic_train_image_caption_path)
epic_train_video_caption_embed = torch.load(epic_train_video_caption_path)
epic_train_mixed_caption_embed = torch.load(epic_train_mixed_caption_path)
epic_csv_action_path_train = "/home/mona/FRIL/FILS/datasets/EK100/epic-kitchens-100-annotations/EPIC_100_train.csv"
generated_csv_path = "/home/mona/FRIL/FILS/datasets/EK100/epic_captions_train.csv"
epic_label = pd.read_csv(epic_csv_action_path_train)
# add another avtion column to epic_label
epic_label['action'] = epic_label['verb'] + ' ' + epic_label['noun']
epic_generated = pd.read_csv(generated_csv_path, delimiter=',')
epic_generated['video'] = epic_generated['video'].apply(lambda x: x.split('/')[-1].split('.')[0].split('_')[-1])


label_dict = {}
image_caption_dict = {}
image_cosine_distance_dict = {}
video_cosine_distance_dict = {}
video_caption_dict = {}
mixed_cosine_distance_dict = {}
mixed_caption_dict = {}
for i in range(len(epic_train_lable_embed)):
    video = f'video_{i}.MP4'
    row = epic_generated.loc[epic_generated['video'] == str(i)]
    label = epic_label['action'][i]
    label_dict[video] = label
    image_caption = row['image_caption'].values[0]
    image_caption_dict[video] = image_caption
    video_caption = row['video_caption'].values[0]
    video_caption_dict[video] = video_caption
    mixed_caption = row['mixed_caption'].values[0]
    mixed_caption_dict[video] = mixed_caption

    image_cosine_distance_dict[video] = cosine_distance(epic_train_lable_embed[str(i)][0], epic_train_image_caption_embed[str(i)]r).item()
    video_cosine_distance_dict[video] = cosine_distance(epic_train_lable_embed[str(i)][0], epic_train_video_caption_embed[str(i)]).item()
    mixed_cosine_distance_dict[video] = cosine_distance(epic_train_lable_embed[str(i)][0], epic_train_mixed_caption_embed[str(i)]).item()
    
    print(video, " Label: ",label, " image_caption:",image_caption, " Cosine distance image_caption & label: ",image_cosine_distance_dict[video],
            " video_caption ",video_caption, " Cosine distance image_caption & label: ",video_cosine_distance_dict[video],
            " mixed_caption ",mixed_caption, " Cosine distance image_caption & label: ",mixed_cosine_distance_dict[video])




#####average distance
# print("Average Euclidean distance: ", np.mean(list(euclidean_distance_dict.values())))
print("Average Cosine distance between image_cations and labels: ", np.mean(list(image_cosine_distance_dict.values())))
print("Average Cosine distance between video_cations and labels: ", np.mean(list(video_cosine_distance_dict.values())))
print("Average Cosine distance between mixed_cations and labels: ", np.mean(list(mixed_cosine_distance_dict.values())))
# # plot the histogram
# plt.hist(euclidean_distance_dict.values(), bins=100)
# plt.title('Euclidean distance')
# plt.xlabel('Distance')
# plt.ylabel('Frequency')
# plt.savefig('euclidean_distance.png')

plt.figure()
plt.hist(image_cosine_distance_dict.values(), bins=100)
plt.title('Cosine distance')
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.savefig('vifi_epic_cosine_distance_bw_image_captions_and_labels.png')

plt.figure()
plt.hist(video_cosine_distance_dict.values(), bins=100)
plt.title('Cosine distance')
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.savefig('vifi_epic_cosine_distance_bw_video_captions_and_labels.png')

plt.figure()
plt.hist(mixed_cosine_distance_dict.values(), bins=100)
plt.title('Cosine distance')
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.savefig('vifi_epic_cosine_distance_bw_mixed_captions_and_labels.png')



# plt.figure()
# plt.hist(dot_product_dict.values(), bins=100)
# plt.title('Dot product')
# plt.xlabel('Distance')
# plt.ylabel('Frequency')
# plt.savefig('dot_product.png')


plt.figure()
plt.hist(image_cosine_distance_dict.values(), bins=100)
plt.hist(video_cosine_distance_dict.values(), bins=100)
plt.hist(mixed_cosine_distance_dict.values(), bins=100)
plt.title('Cosine distance')
plt.xlabel('Distance')
plt.ylabel('Frequency')
plt.legend(['image_caption', 'video_caption', 'mixed_caption'])
plt.savefig('epic_cosine_distance_bw_captions_and_labels.png')


sorted_image_cosine_distance = sorted(image_cosine_distance_dict.items(), key=lambda kv: kv[1])
sorted_video_cosine_distance = sorted(video_cosine_distance_dict.items(), key=lambda kv: kv[1])
sorted_mixed_cosine_distance = sorted(mixed_cosine_distance_dict.items(), key=lambda kv: kv[1])

# print 10 closest and 10 farthest videos'lables and captions
print("10 farthest text embedding based on Cosine distance between image_caption and label: ")
for i in range(10):
    print("Video: ", sorted_image_cosine_distance[i][0], " Label: ", label_dict[sorted_image_cosine_distance[i][0]], " image_Caption: ", image_caption_dict[sorted_image_cosine_distance[i][0]], " video_Caption: ", video_caption_dict[sorted_image_cosine_distance[i][0]], " mixed_Caption: ", mixed_caption_dict[sorted_image_cosine_distance[i][0]])
print("10 closest text embedding based on Cosine distance between image_caption and label: ")
for i in range(len(sorted_image_cosine_distance)-10, len(sorted_image_cosine_distance)):
    print("Video: ", sorted_image_cosine_distance[i][0], " Label: ", label_dict[sorted_image_cosine_distance[i][0]], " image_Caption: ", image_caption_dict[sorted_image_cosine_distance[i][0]], " video_Caption: ", video_caption_dict[sorted_image_cosine_distance[i][0]], " mixed_Caption: ", mixed_caption_dict[sorted_image_cosine_distance[i][0]])

print("10 farthest text embedding based on Cosine distance between video_caption and label: ")
for i in range(10):
    print("Video: ", sorted_video_cosine_distance[i][0], " Label: ", label_dict[sorted_video_cosine_distance[i][0]], " video_Caption: ", video_caption_dict[sorted_video_cosine_distance[i][0]], "image_Caption: ", image_caption_dict[sorted_video_cosine_distance[i][0]], " mixed_Caption: ", mixed_caption_dict[sorted_video_cosine_distance[i][0]])
print("10 closest text embedding based on Cosine distance between video_caption and label: ")
for i in range(len(sorted_video_cosine_distance)-10, len(sorted_video_cosine_distance)):
    print("Video: ", sorted_video_cosine_distance[i][0], " Label: ", label_dict[sorted_video_cosine_distance[i][0]], " video_Caption: ", video_caption_dict[sorted_video_cosine_distance[i][0]], "image_Caption: ", image_caption_dict[sorted_video_cosine_distance[i][0]], " mixed_Caption: ", mixed_caption_dict[sorted_video_cosine_distance[i][0]])

print("10 farthest text embedding based on Cosine distance between mixed_caption and label: ")
for i in range(10):
    print("Video: ", sorted_mixed_cosine_distance[i][0], " Label: ", label_dict[sorted_mixed_cosine_distance[i][0]], " mixed_Caption: ", mixed_caption_dict[sorted_mixed_cosine_distance[i][0]], " image_Caption: ", image_caption_dict[sorted_mixed_cosine_distance[i][0]], " video_Caption: ", video_caption_dict[sorted_mixed_cosine_distance[i][0]])
print("10 closest text embedding based on Cosine distance between mixed_caption and label: ")
for i in range(len(sorted_mixed_cosine_distance)-10, len(sorted_mixed_cosine_distance)):
    print("Video: ", sorted_mixed_cosine_distance[i][0], " Label: ", label_dict[sorted_mixed_cosine_distance[i][0]], " mixed_Caption: ", mixed_caption_dict[sorted_mixed_cosine_distance[i][0]], " image_Caption: ", image_caption_dict[sorted_mixed_cosine_distance[i][0]], " video_Caption: ", video_caption_dict[sorted_mixed_cosine_distance[i][0]])





# ################################################################################################################################## SSV2
# train_label_embed_path_ssv2 = '../../home/mona/SSVLI/dataset/ssv2/ssv2_train_label_text_dict.pt'
# ssv2_train_image_caption_path = '../../home/mona/SSVLI/dataset/ssv2/ssv2_train_image_caption_text_dict.pt'
# ssv2_train_video_caption_path = '../../home/mona/SSVLI/dataset/ssv2/ssv2_train_video_caption_text_dict.pt'
# ssv2_train_mixed_caption_path = '../../home/mona/SSVLI/dataset/ssv2/ssv2_train_mixed_caption_text_dict.pt'

# ssv2_train_label_embed = torch.load(train_label_embed_path_ssv2)
# ssv2_train_image_caption_embed = torch.load(ssv2_train_image_caption_path)
# ssv2_train_video_caption_embed = torch.load(ssv2_train_video_caption_path)
# ssv2_train_mixed_caption_embed = torch.load(ssv2_train_mixed_caption_path)
# ssv2_csv_train_path = "../../home/mona/SSVLI/dataset/ssv2/labels_train.csv"
# ssv2_generated_csv_path = "../../home/mona/SSVLI/dataset/ssv2/ssv2_captions.csv"
# ssv2_label = pd.read_csv(ssv2_csv_train_path, delimiter=' ', header=None)
# ssv2_generated = pd.read_csv(ssv2_generated_csv_path, delimiter=',')
# ssv2_generated['video'] = ssv2_generated['video'].apply(lambda x: x.split('/')[-1].split('.')[0].split('_')[-1])
# # make ssv2_generated['video'] as string
# ssv2_generated['video'] = ssv2_generated['video'].apply(lambda x: str(x))



# label_dict = {}
# image_caption_dict = {}
# image_cosine_distance_dict = {}
# video_cosine_distance_dict = {}
# video_caption_dict = {}
# mixed_cosine_distance_dict = {}
# mixed_caption_dict = {}
# for i in ssv2_label[0]:
#     if str(i) in ssv2_generated['video'].values:
#         video = f'video_{i}.MP4'
#         row = ssv2_generated.loc[ssv2_generated['video'] == str(i)]
#         label = ssv2_label[ssv2_label[0] == i][1].values[0]
#         label_dict[video] = label
#         image_caption = row['image_caption'].values[0]
#         image_caption_dict[video] = image_caption
#         video_caption = row['video_caption'].values[0]
#         video_caption_dict[video] = video_caption
#         mixed_caption = row['mixed_caption'].values[0]
#         mixed_caption_dict[video] = mixed_caption

#         try:
#             image_cosine_distance_dict[video] = cosine_distance(ssv2_train_label_embed[i][0], ssv2_train_image_caption_embed[i][0]).item()
#             video_cosine_distance_dict[video] = cosine_distance(ssv2_train_label_embed[i][0], ssv2_train_video_caption_embed[i][0]).item()
#             mixed_cosine_distance_dict[video] = cosine_distance(ssv2_train_label_embed[i][0], ssv2_train_mixed_caption_embed[i][0]).item()
#         except:
#             a=0
        
#         print(video, " Label: ",label, " image_caption:",image_caption, " Cosine distance image_caption & label: ",image_cosine_distance_dict[video],
#                 " video_caption ",video_caption, " Cosine distance video_caption & label: ",video_cosine_distance_dict[video],
#                 " mixed_caption ",mixed_caption, " Cosine distance mixed_caption & label: ",mixed_cosine_distance_dict[video])




# #####average distance
# # print("Average Euclidean distance: ", np.mean(list(euclidean_distance_dict.values())))
# print("Average Cosine distance between image_cations and labels: ", np.mean(list(image_cosine_distance_dict.values())))
# print("Average Cosine distance between video_cations and labels: ", np.mean(list(video_cosine_distance_dict.values())))
# print("Average Cosine distance between mixed_cations and labels: ", np.mean(list(mixed_cosine_distance_dict.values())))
# # # plot the histogram
# # plt.hist(euclidean_distance_dict.values(), bins=100)
# # plt.title('Euclidean distance')
# # plt.xlabel('Distance')
# # plt.ylabel('Frequency')
# # plt.savefig('euclidean_distance.png')

# plt.figure()
# plt.hist(image_cosine_distance_dict.values(), bins=100)
# plt.title('Cosine distance')
# plt.xlabel('Distance')
# plt.ylabel('Frequency')
# plt.savefig('ssv2_cosine_distance_bw_image_captions_and_labels.png')

# plt.figure()
# plt.hist(video_cosine_distance_dict.values(), bins=100)
# plt.title('Cosine distance')
# plt.xlabel('Distance')
# plt.ylabel('Frequency')
# plt.savefig('ssv2_cosine_distance_bw_video_captions_and_labels.png')

# plt.figure()
# plt.hist(mixed_cosine_distance_dict.values(), bins=100)
# plt.title('Cosine distance')
# plt.xlabel('Distance')
# plt.ylabel('Frequency')
# plt.savefig('ssv2_cosine_distance_bw_mixed_captions_and_labels.png')

# plt.figure()
# plt.hist(image_cosine_distance_dict.values(), bins=100)
# plt.hist(video_cosine_distance_dict.values(), bins=100)
# plt.hist(mixed_cosine_distance_dict.values(), bins=100)
# plt.title('Cosine distance')
# plt.xlabel('Distance')
# plt.ylabel('Frequency')
# plt.legend(['image_caption', 'video_caption', 'mixed_caption'])
# plt.savefig('ssv2_cosine_distance_bw_captions_and_labels.png')


# # plt.figure()
# # plt.hist(dot_product_dict.values(), bins=100)
# # plt.title('Dot product')
# # plt.xlabel('Distance')
# # plt.ylabel('Frequency')
# # plt.savefig('dot_product.png')

# sorted_image_cosine_distance = sorted(image_cosine_distance_dict.items(), key=lambda kv: kv[1])
# sorted_video_cosine_distance = sorted(video_cosine_distance_dict.items(), key=lambda kv: kv[1])
# sorted_mixed_cosine_distance = sorted(mixed_cosine_distance_dict.items(), key=lambda kv: kv[1])

# # print 10 closest and 10 farthest videos'lables and captions
# print("10 farthest text embedding based on Cosine distance between image_caption and label: ")
# for i in range(10):
#     print("Video: ", sorted_image_cosine_distance[i][0], " Label: ", label_dict[sorted_image_cosine_distance[i][0]], " image_Caption: ", image_caption_dict[sorted_image_cosine_distance[i][0]], " video_Caption: ", video_caption_dict[sorted_image_cosine_distance[i][0]], " mixed_Caption: ", mixed_caption_dict[sorted_image_cosine_distance[i][0]])
# print("10 closest text embedding based on Cosine distance between image_caption and label: ")
# for i in range(len(sorted_image_cosine_distance)-10, len(sorted_image_cosine_distance)):
#     print("Video: ", sorted_image_cosine_distance[i][0], " Label: ", label_dict[sorted_image_cosine_distance[i][0]], " image_Caption: ", image_caption_dict[sorted_image_cosine_distance[i][0]], " video_Caption: ", video_caption_dict[sorted_image_cosine_distance[i][0]], " mixed_Caption: ", mixed_caption_dict[sorted_image_cosine_distance[i][0]])

# print("10 farthest text embedding based on Cosine distance between video_caption and label: ")
# for i in range(10):
#     print("Video: ", sorted_video_cosine_distance[i][0], " Label: ", label_dict[sorted_video_cosine_distance[i][0]], " video_Caption: ", video_caption_dict[sorted_video_cosine_distance[i][0]], "image_Caption: ", image_caption_dict[sorted_video_cosine_distance[i][0]], " mixed_Caption: ", mixed_caption_dict[sorted_video_cosine_distance[i][0]])
# print("10 closest text embedding based on Cosine distance between video_caption and label: ")
# for i in range(len(sorted_video_cosine_distance)-10, len(sorted_video_cosine_distance)):
#     print("Video: ", sorted_video_cosine_distance[i][0], " Label: ", label_dict[sorted_video_cosine_distance[i][0]], " video_Caption: ", video_caption_dict[sorted_video_cosine_distance[i][0]], "image_Caption: ", image_caption_dict[sorted_video_cosine_distance[i][0]], " mixed_Caption: ", mixed_caption_dict[sorted_video_cosine_distance[i][0]])

# print("10 farthest text embedding based on Cosine distance between mixed_caption and label: ")
# for i in range(10):
#     print("Video: ", sorted_mixed_cosine_distance[i][0], " Label: ", label_dict[sorted_mixed_cosine_distance[i][0]], " mixed_Caption: ", mixed_caption_dict[sorted_mixed_cosine_distance[i][0]], " image_Caption: ", image_caption_dict[sorted_mixed_cosine_distance[i][0]], " video_Caption: ", video_caption_dict[sorted_mixed_cosine_distance[i][0]])
# print("10 closest text embedding based on Cosine distance between mixed_caption and label: ")
# for i in range(len(sorted_mixed_cosine_distance)-10, len(sorted_mixed_cosine_distance)):
#     print("Video: ", sorted_mixed_cosine_distance[i][0], " Label: ", label_dict[sorted_mixed_cosine_distance[i][0]], " mixed_Caption: ", mixed_caption_dict[sorted_mixed_cosine_distance[i][0]], " image_Caption: ", image_caption_dict[sorted_mixed_cosine_distance[i][0]], " video_Caption: ", video_caption_dict[sorted_mixed_cosine_distance[i][0]])
