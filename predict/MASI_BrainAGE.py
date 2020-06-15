import os
import numpy as np
import argparse
import nibabel as nib
from scipy.io import loadmat


from keras.models import load_model
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def parse_args():
    parser = argparse.ArgumentParser(description="Run dCNN")

    parser.add_argument('--work-dir',required=True, help='Working directory with files needed',type=str)
    parser.add_argument('--output-dir', required=False, help='Output directory (if different from work-dir)', type=str,
                        default=None)

    parser.add_argument('--image-dir',required=False,help='Path to T1 MRI (if different from work-dir)',type=str,default=None)
    parser.add_argument('--features-dir',required=False,help='Path to multi atlas or SLANT volumes (if different from work-dir)',type=str,default=None)

    parser.add_argument('--sex',required=False,help='Sex of the subject (M = 1, F = 2)',type=int,default=None)
    parser.add_argument('--field-strength', required=False, help='Scanner Field Strength', type=float,default=None)



    return parser.parse_args()


def save_estimate(model, image, all_features, out_file):


    out_path = os.path.dirname(out_file)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    all_features=np.expand_dims(all_features,axis=1)
    all_features = np.transpose(all_features)

    y_pred = model.predict([image,all_features])

    print "Predicted age is: ", np.squeeze(y_pred)

    f = open(out_file, 'w')
    f.write("age_pred\n")
    for i in xrange(y_pred.shape[0]):
        try:
            f.write(str(y_pred[i][0]))
        except:
            f.write(str(y_pred[i]))
        f.write("\n")
    f.close()


def load_samples(sample_list):
    samples = []
    if not isinstance(sample_list,list):
        sample_list = [sample_list]

    for sample in xrange(0,len(sample_list)):
        x = nib.load(sample_list[sample])
        samples.append(np.expand_dims(np.squeeze(x.get_data()),axis=0))
    samples = np.concatenate(samples,axis=0)
    samples = np.expand_dims(samples,axis=4)
    return samples


def normalize_features(work_dir, features):

    # Load the training features for rescaling
    file = loadmat(os.path.join(work_dir, 'paths_and_raw_nonorm.mat'))['paths_and_raw_nonorm']
    train_feats = file[:, :]

    # Separate into discrete and continour variables
    train_cat = np.array(train_feats[:, :2])
    train_cont = np.array(train_feats[:, 2:])

    # Standardize to zero-mean and unit variance
    SC = StandardScaler(with_mean=True, with_std=True)
    train_cont = SC.fit_transform(train_cont)
    train = np.concatenate((train_cont, train_cat), axis=1)

    # Rescale to [-1, 1]
    MM = MinMaxScaler(feature_range=(-1, 1))
    train = MM.fit(train)

    # Now do this on the testing features
    test_cat = features[:2]
    test_cont = features[2:]

    test_cont = SC.transform(test_cont)
    test = np.hstack((test_cont, test_cat))
    test = MM.transform(test)

    return test


def main():

    # Inputs
    args = parse_args()
    work_dir = args.work_dir
    output_dir = args.output_dir
    image_dir = args.image_dir
    features_dir = args.features_dir
    sex = args.sex
    field_strength = args.field_strength

    if image_dir is None:
        image_dir = os.path.join(work_dir,'target_processed.nii.gz')
    else:
        image_dir = os.path.join(image_dir, 'target_processed.nii.gz')

    if features_dir is None:
        features_dir = os.path.join(work_dir, 'target_processed_label_volumes.txt')
    else:
        features_dir = os.path.join(features_dir, 'target_processed_label_volumes.txt')

    if output_dir is None:
        output_dir = work_dir

   # The next few lines draw from the distribution of learning data in case sex and field strength are missing.


    print ("Sex and Field Strength not provided. Will sample from training set.")
    if sex is None:
        sex = np.random.choice((1,2),size=1,p=(0.524,0.476))
    if field_strength is None:
        field_strength = np.random.choice((1.5,3),size=1,p=(0.233,0.767))


    # Load Convolutional Model
    print "Loading Trained Convolutional Model"
    model = load_model(os.path.join(work_dir,'BAG_model.h5'))

    # Load Image and Features
    image = load_samples(image_dir)
    features = np.genfromtxt(features_dir, delimiter=',', dtype=float)
    features = features[1:-2,2]

    # concatenate gender, field strength, and multi-atlas features in that order
    all_features = np.hstack((sex,field_strength,features))

    # Normalize features
    all_features = normalize_features(work_dir, all_features)


    # Evaluate Model
    print "Evaluating Model"
    out_file = os.path.join(output_dir,'results.csv')
    save_estimate(model,image,all_features,out_file)
    print "Done"



if __name__=='__main__':
    main()
