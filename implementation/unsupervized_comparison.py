
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from PIL import Image
import random
from tqdm import tqdm
import cv2
import torch
from torch import optim, nn
from torchvision import models, transforms
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from time import time
from sklearn.pipeline import make_pipeline
from sklearn.dummy import DummyClassifier


def load_data(base_path: Path, class_names: list, train: bool, resolution: tuple, randomized: bool, flag: str = "RGB"):
    """_summary_

    Args:
        base_path (Path): Path where the dataset is saved
        class_names (list): list containting the class names. They have to be the same as the folder names of where the images are saved
        train (bool): Whether the method loads the data for train or for test
        resolution (tuple): Images will be resized to this resolution 
        randomized (bool): Wheter to randomized the data after loading or not.
        flag (str, optional): In what format the images will be loaded. Defaults to "RGB".

    Returns:
        (np.array,np.array): Returns preprocessed data and labels.
    """
    data = []
    labels = []
    class_maping = {k:i for i,k in enumerate(class_names)}
    num_images_per_class = 1000 if train else 150
    train = "train" if train else "test"
    print(f"Loading each class for {train}.")
    for class_name in class_names:
        class_folder_path = base_path/train/class_name
        for image_name in tqdm(os.listdir(class_folder_path)[0:num_images_per_class]):
            
            img = np.array(Image.open(class_folder_path/image_name).convert(flag).resize(resolution))

            img_label = class_maping[class_name]
            if img is None:
                    print(f'This image is bad: {class_folder_path/image_name}')
            else:
                labels.append(img_label)
                data.append(img)
    
    if randomized:
        randomized_indices = np.random.permutation(len(labels))
        data, labels = np.array(data)[randomized_indices], np.array(labels)[randomized_indices]
    
    return np.array(data), np.array(labels)


base_path = Path("Z:/Master I/PML - Practical Machine Learning/Unsupervised_Comparison/data/Architectural_Heritage_Elements")

class_names = ['altar', 'apse', 'bell_tower', 'column','dome(inner)','dome(outer)',
               'flying_buttress','gargoyle','stained_glass','vault']


train_data, train_labels = load_data(base_path, class_names, train = True, resolution=(64,64), randomized = True, flag = 'RGB')
# display_some_images(train_data, train_labels)

test_data, test_labels = load_data(base_path, class_names, train = False, resolution = (64,64), randomized = True, flag = 'RGB')
# display_some_images(test_data, test_labels)




def normalize_and_flatten(data: np.array):
    """Method for normalizing and flatten an array.

    Args:
        data (np.array): one image as a numpy array 

    """
    normalized = data / 255.0
    preproc_data = normalized.reshape(len(normalized), -1 )
    return preproc_data


test_features = normalize_and_flatten(test_data)
train_features = normalize_and_flatten(train_data)

# this will be used to extract features using a vgg model 
class FeatureExtractor(nn.Module):
  def __init__(self, model):
    super(FeatureExtractor, self).__init__()

    self.features = list(model.features)
    self.features = nn.Sequential(*self.features)
    self.pooling = model.avgpool
    self.flatten = nn.Flatten()
    self.fc = model.classifier[0]
  
  def forward(self, x):
    out = self.features(x)
    out = self.pooling(out)
    out = self.flatten(out)
    out = self.fc(out) 
    return out 


def extract_features(data, _transform, device):
    """Method for apllying _transform on each image and extract features from it

    Args:
        data (_type_): _description_
        _transform (_type_): PyTorch transform method
        device (_type_): _description_

    Returns:
        Returnes the extracted features from images
    """
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    new_model = FeatureExtractor(model)
    new_model = new_model.to(device)
    features = []
    for i in tqdm(range(len(data))):
        transformed_img = _transform(data[i])
        # if we use vgg as a feature extractor we have to reshape and image to have the channels on the second value
        img = transformed_img.reshape(1, 3, 64, 64)
        img = img.to(device)
        with torch.no_grad():
            feature = new_model(img)
        features.append(feature.cpu().detach().numpy().reshape(-1))

    return np.array(features)


transform = transforms.Compose([
  transforms.ToPILImage(),
  transforms.Resize(64),
  transforms.ToTensor() # this normalizes the data too                       
])


model = models.vgg16(pretrained=True)
new_model = FeatureExtractor(model)


device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
new_model = new_model.to(device)

train_features = extract_features(train_data, transform, device)
test_features = extract_features(test_data, transform, device)


print(test_data.shape)
print(test_labels.shape)
test_features = np.array(test_features)
print(test_features.shape)


print(train_data.shape)
print(train_labels.shape)
train_features = np.array(train_features)
print(train_features.shape)

# saving features as np.arrays so we don't have to extract them everytime
np.save("./data/saved_features/train_features_64_64.npy",train_features)
np.save("./data/saved_features/test_features_64_64.npy",test_features)


def get_labels_mapping(cluster_labels,y_train):
    """
    Associates labels and clusters
    """
    labels_mapping = {}
    nr_of_clusters = len(np.unique(cluster_labels))
    for i in range(nr_of_clusters):
        indexes = np.array([1 if cluster_labels[ind]==i else 0 for ind in range(len(cluster_labels))]) # lista noua cu 1 daca valoarea din cluster e egala cu valoarea lui i si 0 altfel 
        num = np.bincount(y_train[indexes==1]).argmax()
        labels_mapping[i] = num

    return labels_mapping



# Tuning on the number of cluster for kmeans
print(80 * "_")
print("clusters\tinertia\t\thomo\tsilhouette\ttrain_acc\ttest_acc")
for i in [10,15,20,25,50,80,160]:
    # print("Number of clusters: {}".format(i))
    kmeans = KMeans(init="k-means++", n_clusters=i, n_init=4)
    kmeans.fit(train_features)

    _metrics = [i,
                kmeans.inertia_,
                metrics.homogeneity_score(train_labels,kmeans.labels_),
                metrics.silhouette_score(train_features,kmeans.labels_,metric="euclidean")]

    labels_mapping = get_labels_mapping(kmeans.labels_,train_labels)


    number_labels_train = np.zeros(len(kmeans.labels_))
    for i in range(len(kmeans.labels_)):
        number_labels_train[i] = labels_mapping[kmeans.labels_[i]]
    _metrics.append(accuracy_score(number_labels_train,train_labels))


    predicted = kmeans.predict(test_features)
    number_labels_test = np.zeros(len(predicted))
    for i in range(len(predicted)):
        number_labels_test[i] = labels_mapping[predicted[i]]
    _metrics.append(accuracy_score(number_labels_test,test_labels))

    formatter_result = ("{}\t\t{:.3f}\t{:.3f}\t{:.3f}\t\t{:.3f}\t\t{:.3f}")

    print(formatter_result.format(*_metrics))
print(80 * "_")


# We have rewrited the code in a method so that we can use it more easily
def tune_k_means(kmeans, name, data, labels):

    starting_time = time()
    estimator = make_pipeline(StandardScaler(), kmeans).fit(data)
    fit_time = time() - starting_time

    _metrics = [name,
                fit_time,
                estimator[-1].inertia_,
                metrics.homogeneity_score(labels, estimator[-1].labels_),
                metrics.silhouette_score(data,
                                         estimator[-1].labels_,
                                         metric="euclidean",
                                         sample_size=300,)]

    # Print the results
    formatter_result = ("{:9s}\t{:.3f}s\t{:.3f}\t{:.3f}\t{:.3f}")
    print(formatter_result.format(*_metrics))


print(60 * "_")
print("init\t\ttime\tinertia\t\thomo\tsilhouette")

kmeans = KMeans(init="k-means++", n_clusters=25, n_init=4, random_state=0)
tune_k_means(kmeans=kmeans, name="k-means++", data=train_features, labels=train_labels)

kmeans = KMeans(init="random", n_clusters=25, n_init=4, random_state=0)
tune_k_means(kmeans=kmeans, name="random", data=train_features, labels=train_labels)

pca = PCA(n_components=25).fit(train_features)
kmeans = KMeans(init=pca.components_, n_clusters=25, n_init=1)
tune_k_means(kmeans=kmeans, name="PCA-based", data=train_features, labels=train_labels)

print(60  * "_")


# Tuning the algorithm that kmeans uses
def tune_k_means(kmeans, name, data, labels):

    starting_time = time()
    kmeans.fit(data)
    fit_time = time() - starting_time

    _metrics = [name,
                fit_time,
                kmeans.inertia_,
                metrics.homogeneity_score(labels, kmeans.labels_),
                metrics.silhouette_score(data,
                                         kmeans.labels_,
                                         metric="euclidean",
                                         sample_size=300,)]

    labels_mapping = get_labels_mapping(kmeans.labels_,train_labels)


    number_labels_train = np.zeros(len(kmeans.labels_))
    for i in range(len(kmeans.labels_)):
        number_labels_train[i] = labels_mapping[kmeans.labels_[i]]
    _metrics.append(accuracy_score(number_labels_train,train_labels))

    predicted = kmeans.predict(test_features)
    number_labels_test = np.zeros(len(predicted))
    for i in range(len(predicted)):
        number_labels_test[i] = labels_mapping[predicted[i]]
    _metrics.append(accuracy_score(number_labels_test,test_labels))

    formatter_result = ("{:.1}\t\t{:.3f}\t{:.3f}\t{:.3f}\t\t{:.3f}\t\t{:.3f}")
    
    # Print the results
    formatter_result = ("{:9s}\t{:.3f}s\t{:.3f}\t{:.3f}\t{:.3f}\t\t{:.3f}\t\t{:.3f}")
    print(formatter_result.format(*_metrics))


print(90 * "_")
print("algorithm\ttime\tinertia\t\thomo\tsilhouette\tacc_train\tacc_test")

kmeans = KMeans(init="k-means++", n_clusters=25, n_init=4, random_state=0, algorithm="lloyd")
tune_k_means(kmeans=kmeans, name="lloyd", data=train_features, labels=train_labels)

kmeans = KMeans(init="k-means++", n_clusters=25, n_init=4, random_state=0,algorithm="elkan")
tune_k_means(kmeans=kmeans, name="elkan", data=train_features, labels=train_labels)


print(90 * "_")


# Method for dbscan tuning
def tune_db_scan(dbscan, value, data, labels):

    starting_time = time()
    estimator = make_pipeline(StandardScaler(), dbscan).fit(data)
    fit_time = time() - starting_time

    _metrics = [value,
                fit_time,
                metrics.silhouette_score(data,
                                         estimator[-1].labels_),
                metrics.homogeneity_score(labels, estimator[-1].labels_),
                metrics.completeness_score(labels, estimator[-1].labels_),
                len(np.unique(dbscan.labels_))]

    # Print the results
    formatter_result = ("{}\t\t{:.3f}\t{:.3f}\t\t{:.3f}\t{:.3f}\t{}")
    print(formatter_result.format(*_metrics))

# Tuning eps with min_samples = 3
print(80 * "_")
print("eps_val\t\ttime\tsilhouette\thomo\tcomplet\tnr_clusters")

for i in [10,20,30,40,60,80]:
    db = DBSCAN(eps=i, min_samples=3)
    tune_db_scan(dbscan=db, value=i, data=train_features, labels=train_labels)

print(80 * "_")


# Tuning min_samples with eps = 40
print(80 * "_")
print("min_samples\ttime\tsilhouette\thomo\tcomplet\tnr_clusters")

for i in [2,3,4,5,6,7]:
    db = DBSCAN(eps=40, min_samples=i)
    tune_db_scan(dbscan=db, value=i, data=train_features, labels=train_labels)

print(80 * "_")

# Tuning min_samples with eps = 60
print(80 * "_")
print("min_samples\ttime\tsilhouette\thomo\tcomplet\tnr_clusters")

for i in [2,3,4,5,6,7]:
    db = DBSCAN(eps=60, min_samples=i)
    tune_db_scan(dbscan=db, value=i, data=train_features, labels=train_labels)

print(80 * "_")

# Tuning min_samples with eps = 80
print(80 * "_")
print("min_samples\ttime\tsilhouette\thomo\tcomplet\tnr_clusters")

for i in [2,3,4,5,6,7]:
    db = DBSCAN(eps=80, min_samples=i)
    tune_db_scan(dbscan=db, value=i, data=train_features, labels=train_labels)

print(80 * "_")


# Comparing kmeans dbscan random and SVM

svm = SVC(kernel= 'linear')
svm.fit(train_features, train_labels)
y_pred = svm.predict(test_features)
print(accuracy_score(test_labels, y_pred))


dummy_clf = DummyClassifier(strategy="most_frequent")
dummy_clf.fit(train_features, train_labels)
y_pred = dummy_clf.predict(test_features)
print(accuracy_score(test_labels, y_pred))


def results(algo):


    if algo == "random":
        starting_time = time()
        dummy_clf = DummyClassifier(strategy="most_frequent")
        dummy_clf.fit(train_features, train_labels)
        fit_time = time() - starting_time
        y_pred_test = dummy_clf.predict(test_features)
        y_pred_train = dummy_clf.predict(train_features)

        _metrics = [algo,
                    fit_time,
                    "none",
                    accuracy_score(train_labels, y_pred_train),
                    accuracy_score(test_labels, y_pred_test)]
    elif algo == "SVM":
        starting_time = time()
        svm = SVC(kernel= 'linear')
        svm.fit(train_features, train_labels)
        fit_time = time() - starting_time
        y_pred_test = svm.predict(test_features)
        y_pred_train = svm.predict(train_features)
        _metrics = [algo,
                    fit_time,
                    "none",
                    accuracy_score(train_labels, y_pred_train),
                    accuracy_score(test_labels, y_pred_test)]
    elif algo == "kmeans":
        starting_time = time()
        kmeans = KMeans(init="k-means++", n_clusters=25, n_init=4)
        kmeans.fit(train_features)
        fit_time = time() - starting_time
        labels_mapping = get_labels_mapping(kmeans.labels_,train_labels)

        number_labels_train = np.zeros(len(kmeans.labels_))
        for i in range(len(kmeans.labels_)):
            number_labels_train[i] = labels_mapping[kmeans.labels_[i]]

        predicted = kmeans.predict(test_features)
        number_labels_test = np.zeros(len(predicted))
        for i in range(len(predicted)):
            number_labels_test[i] = labels_mapping[predicted[i]]

        _metrics =[algo,
                  fit_time,
                  metrics.silhouette_score(train_features,kmeans.labels_,metric="euclidean"),
                  accuracy_score(number_labels_train,train_labels),
                  accuracy_score(number_labels_test,test_labels)]

    elif algo == "dbscan":
        starting_time = time()
        estimator = make_pipeline(StandardScaler(), DBSCAN(eps=80, min_samples=5)).fit(train_features)
        fit_time = time() - starting_time

        _metrics = [algo,
                    fit_time,
                    metrics.silhouette_score(train_features,
                                             estimator[-1].labels_),
                    "none",
                    "none"]

    formatter_result = ("{}\t\t{}\t{}\t\t\t{}\t\t{}")
    print(formatter_result.format(*_metrics))


print(120 * "_")
print("algorithm\ttime\t\t\tsilhouette\t\t\tacc_train\t\t\tacc_test")

results("random")
results("SVM")
results("kmeans")
results("dbscan")
print(120 * "_")


