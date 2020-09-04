from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from datasets.simpledatasetloader import SimpleDatasetLoader
from preprocessing.simplepreprocessor import SimplePreprocessor
from imutils import paths
import argparse

# pare arguments
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-d', '--dataset', required=True, help='Path to input dataset')
arg_parser.add_argument('-k', '--neighbours', type=int, default=1,
                       help='Number of nearest neighbours for classification')
arg_parser.add_argument('-j', '--jobs', type=int, default=1,
                        help='Number of jobs for k-NN distance (-1 uses all available cores)')
args = vars(arg_parser.parse_args())

# Load the dataset
print('[INFO] loading images...')
image_paths = list(paths.list_images(args['dataset']))
resize_preprocessor = SimplePreprocessor(32, 32)
dataset_loader = SimpleDatasetLoader(preprocessors=[resize_preprocessor])
data, labels = dataset_loader.load(image_paths, verbose=500)
data = data.reshape((data.shape[0], -1))
print(f'[INFO] features matrix {data.nbytes / 1024 * 1000:1f}MB')

# Encode labels to integers
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Splitting the dataset
train_X, test_X, train_y, test_y = train_test_split(data, labels, test_size=0.2)

# Evaluating the Model
model = KNeighborsClassifier(n_neighbors=args['neighbours'], metric='cosine', algorithm='brute', n_jobs=args['jobs'])
model.fit(train_X, train_y)
pred_y = model.predict(test_X)
print(classification_report(test_y, pred_y, target_names=label_encoder.classes_))
