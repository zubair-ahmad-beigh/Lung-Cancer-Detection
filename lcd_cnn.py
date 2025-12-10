from codecs import BOM32_BE
from ctypes import alignment
from unittest import result
from xml.dom.expatbuilder import parseString
import numpy as np
import pandas as pd
import pydicom as dicom
import os
import matplotlib.pyplot as plt
import cv2
import math

import tensorflow._api.v2.compat.v1 as tf 
tf.disable_v2_behavior()
import pandas as pd
"""Removed tflearn imports as they are unused and can break with TF2."""
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

from tkinter import *
from tkinter import messagebox,ttk,filedialog
import tkinter as tk
from PIL import Image,ImageTk


class LCD_CNN:
    def __init__(self,root):
        self.root=root
        #window size
        self.root.geometry("1006x500+0+0")
        self.root.resizable(False, False)
        self.root.title("Lung Cancer Detection")
        # Ensure the window shows on top initially
        try:
            self.root.attributes('-topmost', True)
            self.root.update()
            self.root.attributes('-topmost', False)
            self.root.focus_force()
        except Exception:
            pass

        try:
            img4=Image.open(r"Images\Lung-Cancer-Detection.jpg")
            # Pillow 10+: ANTIALIAS moved to Resampling.LANCZOS
            img4=img4.resize((1006,500), Image.Resampling.LANCZOS)
            #Antialiasing is a technique used in digital imaging to reduce the visual defects that occur when high-resolution images are presented in a lower resolution.
            self.photoimg4=ImageTk.PhotoImage(img4)

            bg_img=Label(self.root,image=self.photoimg4)
            bg_img.place(x=0,y=50,width=1006,height=500)
        except Exception as e:
            # Fallback: solid background if image fails
            bg_img=Label(self.root, bg="lightgray")
            bg_img.place(x=0,y=50,width=1006,height=500)

        # title Label
        title_lbl=Label(text="Lung Cancer Detection",font=("Bradley Hand ITC",30,"bold"),bg="black",fg="white",)
        title_lbl.place(x=0,y=0,width=1006,height=50)

        #button 1
        self.b1=Button(text="Import Data",cursor="hand2",command=self.import_data,font=("Times New Roman",15,"bold"),bg="white",fg="black")
        self.b1.place(x=80,y=130,width=180,height=30)

        #button 2
        self.b2=Button(text="Pre-Process Data",cursor="hand2",command=self.preprocess_data,font=("Times New Roman",15,"bold"),bg="white",fg="black")
        self.b2.place(x=80,y=180,width=180,height=30)
        self.b2["state"] = "disabled"
        self.b2.config(cursor="arrow")

        #button 3
        self.b3=Button(text="Train Data",cursor="hand2",command=self.train_data,font=("Times New Roman",15,"bold"),bg="white",fg="black")
        self.b3.place(x=80,y=230,width=180,height=30)
        self.b3["state"] = "disabled"
        self.b3.config(cursor="arrow")

        #button 4 - Predict Image
        self.b4=Button(text="Predict Image",cursor="hand2",command=self.predict_image,font=("Times New Roman",15,"bold"),bg="lightgreen",fg="black")
        self.b4.place(x=80,y=280,width=180,height=30)
        self.b4["state"] = "disabled"
        self.b4.config(cursor="arrow")

        # If preprocessed file already exists, enable training directly
        try:
            if os.path.exists('imageDataNew-10-10-5.npy'):
                self.b3["state"] = "normal"
                self.b3.config(cursor="hand2")
            # If model exists, enable prediction
            if os.path.exists('lung_cancer_model.ckpt.meta') or os.path.exists('lung_cancer_model'):
                self.b4["state"] = "normal"
                self.b4.config(cursor="hand2")
        except Exception:
            pass
        
        # Store model session and graph for prediction
        self.trained_session = None
        self.trained_prediction = None
        self.trained_x = None

#Data Import lets you upload data from external sources and combine it with data you collect via Analytics.
    def import_data(self):
        # Open folder selection dialog
        folder_path = filedialog.askdirectory(
            title="Select Images Folder (e.g., ToyDataset or sample_images)",
            initialdir=os.getcwd()
        )
        
        if not folder_path:
            messagebox.showwarning("Import Data", "No folder selected. Please select a folder containing images.")
            return
        
        # Set data directory to selected folder (normalize path)
        self.dataDirectory = os.path.normpath(folder_path) + os.sep
        
        # Check if folder exists and has content
        if not os.path.exists(self.dataDirectory):
            messagebox.showerror("Error", f"Selected folder does not exist: {self.dataDirectory}")
            return
        
        try:
            # Get all items in the directory
            all_items = os.listdir(self.dataDirectory)
            
            # Filter to get only directories (subfolders)
            self.lungPatients = [d for d in all_items if os.path.isdir(os.path.join(self.dataDirectory, d))]
            
            # Also check if there are image files directly in the folder
            image_files = [f for f in all_items if os.path.isfile(os.path.join(self.dataDirectory, f)) and 
                          f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.dcm', '.dicom'))]
            
            # Debug info
            print(f"Selected folder: {self.dataDirectory}")
            print(f"Found {len(self.lungPatients)} subfolders: {self.lungPatients}")
            print(f"Found {len(image_files)} image files directly in folder")
            
            # If no subfolders and no images, show error
            if len(self.lungPatients) == 0 and len(image_files) == 0:
                messagebox.showerror("Error", 
                    f"No subfolders or images found in:\n{self.dataDirectory}\n\n"
                    f"Please ensure the folder contains:\n"
                    f"- Subfolders like: aca/, normal/, scc/, benign/, malignant/\n"
                    f"- OR image files directly in the folder")
                return
            
            # If only images found (no subfolders), create a temporary structure
            if len(self.lungPatients) == 0 and len(image_files) > 0:
                # Treat the folder itself as a single patient
                folder_name = os.path.basename(os.path.normpath(folder_path))
                self.lungPatients = [folder_name]
                # Adjust dataDirectory to parent
                self.dataDirectory = os.path.dirname(self.dataDirectory) + os.sep
                print(f"Found images directly, treating as single patient: {folder_name}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error reading folder: {str(e)}\n\nPath: {self.dataDirectory}")
            import traceback
            traceback.print_exc()
            return

        ##Read labels csv 
        try:
            self.labels = pd.read_csv('stage1_labels.csv', index_col=0)
        except FileNotFoundError:
            # If CSV doesn't exist, create empty DataFrame
            self.labels = pd.DataFrame()
            print("CSV file not found. Will use folder names to infer labels.")

        ##Setting x*y size to 10
        self.size = 10

        ## Setting z-dimension (number of slices to 5)
        self.NoSlices = 5

        # Create success message
        folders_list = ', '.join(self.lungPatients[:10])
        if len(self.lungPatients) > 10:
            folders_list += f' ... and {len(self.lungPatients) - 10} more'
        
        messagebox.showinfo("Import Data" , 
            f"Data Imported Successfully!\n\n"
            f"Selected Folder: {os.path.basename(folder_path)}\n"
            f"Full Path: {folder_path}\n"
            f"Found {len(self.lungPatients)} subfolder(s):\n{folders_list}\n\n"
            f"Click 'Pre-Process Data' to continue.") 

        self.b1["state"] = "disabled"
        self.b1.config(cursor="arrow") 
        self.b2["state"] = "normal"
        self.b2.config(cursor="hand2")   

# Data preprocessing is the process of transforming raw data into an understandable format.
    def preprocess_data(self):

        def chunks(l, n):
            count = 0
            for i in range(0, len(l), n):
                if (count < self.NoSlices):
                    yield l[i:i + n]
                    count = count + 1


        def mean(l):
            return sum(l) / len(l)
        #Average


        def dataProcessing(patient, labels_df, size=10, noslices=5, visualize=False):
            # Try to get label from CSV, if not found, infer from directory name
            try:
                label = labels_df._get_value(patient, 'cancer')
            except (KeyError, AttributeError):
                # If label not in CSV, check directory name
                patient_lower = patient.lower()
                if 'benign' in patient_lower or 'normal' in patient_lower:
                    label = 0  # No cancer
                elif 'malignant' in patient_lower or 'cancer' in patient_lower or 'aca' in patient_lower or 'scc' in patient_lower:
                    label = 1  # Cancer (adenocarcinoma, squamous cell carcinoma, or malignant)
                else:
                    label = 0  # Default to non-cancer
            
            path = os.path.join(self.dataDirectory, patient)
            slices = []
            
            # Check if path exists
            if not os.path.exists(path):
                print(f'Path does not exist: {path}')
                return None, None
            
            # Get all files in the directory
            try:
                all_items = os.listdir(path)
                files = [f for f in all_items if os.path.isfile(os.path.join(path, f))]
            except Exception as e:
                print(f'Error listing directory {path}: {e}')
                return None, None
            
            # Process DICOM files or regular image files
            for file in files:
                file_path = os.path.join(path, file)
                try:
                    if file.lower().endswith(('.dcm', '.dicom')):
                        # Handle DICOM files
                        dicom_file = dicom.read_file(file_path)
                        img_array = np.array(dicom_file.pixel_array)
                    elif file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        # Handle regular image files
                        img_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                        if img_array is None:
                            continue
                    else:
                        continue
                    
                    slices.append(img_array)
                except Exception as e:
                    print(f'Error processing {file_path}: {e}')
                    continue
            
            # If no slices found, return None
            if len(slices) == 0:
                return None, None

            new_slices = []
            # Resize all slices to the required size
            resized_slices = []
            for slice_img in slices:
                if len(slice_img.shape) == 2:
                    # Already grayscale
                    resized = cv2.resize(slice_img, (size, size))
                else:
                    # Convert to grayscale if colored
                    gray = cv2.cvtColor(slice_img, cv2.COLOR_BGR2GRAY)
                    resized = cv2.resize(gray, (size, size))
                resized_slices.append(resized)
            
            slices = resized_slices

            # If we have fewer slices than needed, duplicate the last slice
            while len(slices) < noslices:
                slices.append(slices[-1] if slices else np.zeros((size, size)))

            # Ensure we have at least noslices
            if len(slices) > 0:
                chunk_sizes = math.floor(len(slices) / noslices) if len(slices) >= noslices else 1
                if chunk_sizes == 0:
                    chunk_sizes = 1
                    
            for slice_chunk in chunks(slices, chunk_sizes):
                    if len(slice_chunk) > 0:
                        # Average the chunk into a single slice
                        if len(slice_chunk) == 1:
                            # If only one slice in chunk, use it directly
                            averaged_slice = slice_chunk[0]
                        else:
                            # Average multiple slices
                            averaged_slice = np.mean(slice_chunk, axis=0)
                        new_slices.append(averaged_slice)
                    
                    # Stop if we have enough slices
                    if len(new_slices) >= noslices:
                        break

            # Ensure we have exactly noslices with consistent shape
            while len(new_slices) < noslices:
                if len(new_slices) > 0:
                    new_slices.append(new_slices[-1].copy())
                else:
                    new_slices.append(np.zeros((size, size)))
            
            # Truncate if we have more than needed
            new_slices = new_slices[:noslices]
            
            # Ensure all slices have the correct shape
            final_slices = []
            for s in new_slices:
                if isinstance(s, np.ndarray) and s.shape == (size, size):
                    final_slices.append(s)
                else:
                    final_slices.append(np.zeros((size, size)))
            
            # Convert to numpy array with proper shape
            # Ensure all slices are numpy arrays with shape (size, size)
            validated_slices = []
            for s in new_slices:
                if isinstance(s, np.ndarray):
                    if s.shape == (size, size):
                        validated_slices.append(s)
                    else:
                        # Resize if shape is wrong
                        validated_slices.append(cv2.resize(s, (size, size)))
                else:
                    validated_slices.append(np.zeros((size, size)))
            
            # Convert to numpy array - should be shape (noslices, size, size)
            final_array = np.array(validated_slices, dtype=np.float32)
            
            # Ensure shape is correct: (noslices, size, size) = (5, 10, 10)
            if final_array.shape != (noslices, size, size):
                print(f'Warning: Shape mismatch. Expected ({noslices}, {size}, {size}), got {final_array.shape}')
                # Reshape if needed
                if len(final_array.shape) == 2:
                    # If it's 2D, expand dimensions
                    final_array = np.expand_dims(final_array, 0)
                    # Duplicate to get noslices
                    while final_array.shape[0] < noslices:
                        final_array = np.vstack([final_array, final_array[0:1]])
                    final_array = final_array[:noslices]
                final_array = final_array.reshape(noslices, size, size)
            
            # Keep shape as (5, 10, 10) - will be reshaped in the model
            # Model expects input that can be reshaped to (batch, 10, 10, 5, 1)

            if label == 1: #Cancer Patient
                label = np.array([0, 1])
            elif label == 0:    #Non Cancerous Patient
                label = np.array([1, 0])
            return final_array, label


        imageData = []
        #Check if Data Labels is available in CSV or not
        for num, patient in enumerate(self.lungPatients):
            if num % 50 == 0:
                print('Saved -', num)
            try:
                # Skip if it's not a directory (like .DS_Store files)
                patient_path = os.path.join(self.dataDirectory, patient)
                if not os.path.isdir(patient_path):
                    continue
                    
                img_data, label = dataProcessing(patient, self.labels, size=self.size, noslices=self.NoSlices)
                
                # Skip if processing failed
                if img_data is None or label is None:
                    print(f'Skipping {patient}: No valid images found')
                    continue
                    
                # Verify shapes before appending
                if img_data.shape != (self.NoSlices, self.size, self.size):
                    print(f'Warning: {patient} has wrong shape {img_data.shape}, expected ({self.NoSlices}, {self.size}, {self.size})')
                    continue
                
                # Transpose to (height, width, slices) for model: from (5, 10, 10) to (10, 10, 5)
                img_data = img_data.transpose(1, 2, 0)
                    
                imageData.append([img_data, label, patient])
                print(f'Processed {patient}: shape={img_data.shape}, label={label}')
            except KeyError as e:
                print(f'Data is unlabeled for {patient}: {e}')
            except Exception as e:
                print(f'Error processing {patient}: {e}')
                import traceback
                traceback.print_exc()


        ##Results= Image Data and lable.
        if len(imageData) == 0:
            messagebox.showerror("Error", "No data processed! Please check your ToyDataset folder.")
            return
            
        print(f'Total processed: {len(imageData)} patients')
        # Save using pickle format to handle mixed data types
        import pickle
        with open('imageDataNew-{}-{}-{}.npy'.format(self.size, self.size, self.NoSlices), 'wb') as f:
            pickle.dump(imageData, f)
        print('Data saved successfully!')

        messagebox.showinfo("Pre-Process Data" , "Data Pre-Processing Done Successfully!") 

        self.b2["state"] = "disabled"
        self.b2.config(cursor="arrow") 
        self.b3["state"] = "normal"
        self.b3.config(cursor="hand2")

# Data training is the process of training the model based on the dataset and then predict on new data.
    def train_data(self):    

        # Load data using pickle
        import pickle
        with open('imageDataNew-10-10-5.npy', 'rb') as f:
            imageData = pickle.load(f)
        
        print(f'Total data loaded: {len(imageData)} patients')
        
        # Dynamically split data based on available patients
        total_patients = len(imageData)
        if total_patients >= 5:
            # If we have enough data, use standard split
            split_point = int(total_patients * 0.9)  # 90% training, 10% validation
            trainingData = imageData[:split_point]
            validationData = imageData[split_point:]
        else:
            # If we have less data, use all for training and duplicate for validation
            trainingData = imageData
            validationData = imageData[:1] if total_patients > 1 else [imageData[0]]
            print(f'Note: Using all {total_patients} patient(s) for training, repeating for validation')

        training_data=Label(text="Total Training Data: " + str(len(trainingData)),font=("Times New Roman",13,"bold"),bg="black", fg="white",)
        training_data.place(x=750,y=150,width=200,height=18)   

        validation_data=Label(text="Total Validation Data: " + str(len(validationData)),font=("Times New Roman",13,"bold"),bg="black",fg="white",)
        validation_data.place(x=750,y=190,width=200,height=18)  

        x = tf.placeholder('float')
        y = tf.placeholder('float')
        size = 10
        keep_rate = 0.8
        NoSlices = 5

        def convolution3d(x, W):
            return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')


        def maxpooling3d(x):
            return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

        def cnn(x):
            x = tf.reshape(x, shape=[-1, size, size, NoSlices, 1])
            convolution1 = tf.nn.relu(
                convolution3d(x, tf.Variable(tf.random_normal([3, 3, 3, 1, 32]))) + tf.Variable(tf.random_normal([32])))
            convolution1 = maxpooling3d(convolution1)
            convolution2 = tf.nn.relu(
                convolution3d(convolution1, tf.Variable(tf.random_normal([3, 3, 3, 32, 64]))) + tf.Variable(
                    tf.random_normal([64])))
            convolution2 = maxpooling3d(convolution2)
            convolution3 = tf.nn.relu(
                convolution3d(convolution2, tf.Variable(tf.random_normal([3, 3, 3, 64, 128]))) + tf.Variable(
                    tf.random_normal([128])))
            convolution3 = maxpooling3d(convolution3)
            convolution4 = tf.nn.relu(
                convolution3d(convolution3, tf.Variable(tf.random_normal([3, 3, 3, 128, 256]))) + tf.Variable(
                    tf.random_normal([256])))
            convolution4 = maxpooling3d(convolution4)
            convolution5 = tf.nn.relu(
                convolution3d(convolution4, tf.Variable(tf.random_normal([3, 3, 3, 256, 512]))) + tf.Variable(
                    tf.random_normal([512])))
            # Correct pooling: pool the result of the fifth conv, not the fourth
            convolution5 = maxpooling3d(convolution5)

            # Use Keras layers for TF 2.x + Keras 3 compatibility
            flattened = tf.keras.layers.Flatten()(convolution5)
            fullyconnected = tf.keras.layers.Dense(256, activation=tf.nn.relu)(flattened)
            fullyconnected = tf.nn.dropout(fullyconnected, keep_prob=keep_rate)
            output = tf.keras.layers.Dense(2)(fullyconnected)
            return output

        def network(x):
            prediction = cnn(x)
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)
            epochs = 100
            saver = tf.train.Saver()
            with tf.Session() as session:
                session.run(tf.global_variables_initializer())
                for epoch in range(epochs):
                    epoch_loss = 0
                    for data in trainingData:
                        try:
                            X = data[0]
                            Y = data[1]
                            _, c = session.run([optimizer, cost], feed_dict={x: X, y: Y})
                            epoch_loss += c
                        except Exception as e:
                            pass
                        
                    correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
                   # if tf.argmax(prediction, 1) == 0:
                    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
                    print('Epoch', epoch + 1, 'completed out of', epochs, 'loss:', epoch_loss)
                    # print('Correct:',correct.eval({x:[i[0] for i in validationData], y:[i[1] for i in validationData]}))
                    print('Accuracy:', accuracy.eval({x: [i[0] for i in validationData], y: [i[1] for i in validationData]}))
                #print('Final Accuracy:', accuracy.eval({x: [i[0] for i in validationData], y: [i[1] for i in validationData]}))
                x1 = accuracy.eval({x: [i[0] for i in validationData], y: [i[1] for i in validationData]})

                final_accuracy=Label(text="Final Accuracy: " + str(x1),font=("Times New Roman",13,"bold"),bg="black", fg="white",)
                final_accuracy.place(x=750,y=230,width=200,height=18)  

                patients = []
                actual = []
                predicted = []

                finalprediction = tf.argmax(prediction, 1)
                actualprediction = tf.argmax(y, 1)
                for i in range(len(validationData)):
                    patients.append(validationData[i][2])
                for i in finalprediction.eval({x: [i[0] for i in validationData], y: [i[1] for i in validationData]}):
                    if(i==1):
                        predicted.append("Cancer")
                    else:
                        predicted.append("No Cancer")
                for i in actualprediction.eval({x: [i[0] for i in validationData], y: [i[1] for i in validationData]}):
                    if(i==1):
                        actual.append("Cancer")
                    else:
                        actual.append("No Cancer")
                for i in range(len(patients)):
                    print("----------------------------------------------------")
                    print("Patient: ",patients[i])
                    print("Actual: ", actual[i])
                    print("Predicted: ", predicted[i])
                    print("----------------------------------------------------")

                # messagebox.showinfo("Result" , "Patient: " + ' '.join(map(str,patients)) + "\nActual: " + str(actual) + "\nPredicted: " + str(predicted) + "Accuracy: " + str(x1))    

                y_actual = pd.Series(
                    (actualprediction.eval({x: [i[0] for i in validationData], y: [i[1] for i in validationData]})),
                    name='Actual')
                y_predicted = pd.Series(
                    (finalprediction.eval({x: [i[0] for i in validationData], y: [i[1] for i in validationData]})),
                    name='Predicted')

                df_confusion = pd.crosstab(y_actual, y_predicted).reindex(columns=[0,1],index=[0,1], fill_value=0)
                print('Confusion Matrix:\n')
                print(df_confusion)

                prediction_label=Label(text=">>>>    P R E D I C T I O N    <<<<",font=("Times New Roman",14,"bold"),bg="#778899", fg="black",)
                prediction_label.place(x=0,y=458,width=1006,height=20)   

                result1 = []

                for i in range(len(validationData)):
                    result1.append(patients[i])
                    if(y_actual[i] == 1):
                        result1.append("Cancer")
                    else:
                        result1.append("No Cancer")

                    if(y_predicted[i] == 1):
                        result1.append("Cancer")
                    else:
                        result1.append("No Cancer")

                # print(result1)

                total_rows = int(len(patients))
                total_columns = int(len(result1)/len(patients))  

                heading = ["Patient: ", "Actual: ", "Predicted: "]

                self.root.geometry("1006x"+str(500+(len(patients)*20)-20)+"+0+0") 
                self.root.resizable(False, False)

                for i in range(total_rows):
                    for j in range(total_columns):
                 
                        self.e = Entry(root, width=42, fg='black', font=('Times New Roman',12,'bold')) 
                        self.e.grid(row=i, column=j) 
                        self.e.place(x=(j*335),y=(478+i*20))
                        self.e.insert(END, heading[j] + result1[j + i*3]) 
                        self.e["state"] = "disabled"
                        self.e.config(cursor="arrow")                     

                # Save the trained model before closing session
                try:
                    save_path = saver.save(session, 'lung_cancer_model.ckpt')
                    print(f'Model saved to {save_path}')
                    # Enable predict button
                    self.b4["state"] = "normal"
                    self.b4.config(cursor="hand2")
                except Exception as e:
                    print(f'Error saving model: {e}')

                self.b3["state"] = "disabled"
                self.b3.config(cursor="arrow") 

                messagebox.showinfo("Train Data" , "Model Trained Successfully!")

                ## Function to plot confusion matrix
                def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):\

                    plt.matshow(df_confusion, cmap=cmap)  # imshow  
                    # plt.title(title)
                    plt.colorbar()
                    tick_marks = np.arange(len(df_confusion.columns))
                    plt.title(title)
                    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
                    plt.yticks(tick_marks, df_confusion.index)
                    # plt.tight_layout()
                    plt.ylabel(df_confusion.index.name)
                    plt.xlabel(df_confusion.columns.name)
                    plt.show()
                plot_confusion_matrix(df_confusion)
                # print(y_true,y_pred)
                # print(confusion_matrix(y_true, y_pred))
                # print(actualprediction.eval({x:[i[0] for i in validationData], y:[i[1] for i in validationData]}))
                # print(finalprediction.eval({x:[i[0] for i in validationData], y:[i[1] for i in validationData]}))  

        network(x)

    # Function to preprocess a single image for prediction
    def preprocess_single_image(self, image_path, size=10, noslices=5):
        """Preprocess a single image file for prediction"""
        slices = []
        
        # Check if it's a directory (multiple images) or single file
        if os.path.isdir(image_path):
            # Get all files in the directory
            files = [f for f in os.listdir(image_path) if os.path.isfile(os.path.join(image_path, f))]
            for file in files:
                file_path = os.path.join(image_path, file)
                try:
                    if file.lower().endswith(('.dcm', '.dicom')):
                        dicom_file = dicom.read_file(file_path)
                        img_array = np.array(dicom_file.pixel_array)
                    elif file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                        img_array = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                        if img_array is None:
                            continue
                    else:
                        continue
                    slices.append(img_array)
                except Exception as e:
                    print(f'Error processing {file_path}: {e}')
                    continue
        else:
            # Single image file
            try:
                if image_path.lower().endswith(('.dcm', '.dicom')):
                    dicom_file = dicom.read_file(image_path)
                    img_array = np.array(dicom_file.pixel_array)
                elif image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    img_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    if img_array is None:
                        return None
                else:
                    return None
                slices.append(img_array)
            except Exception as e:
                print(f'Error processing {image_path}: {e}')
                return None
        
        if len(slices) == 0:
            return None
        
        # Resize all slices
        resized_slices = []
        for slice_img in slices:
            if len(slice_img.shape) == 2:
                resized = cv2.resize(slice_img, (size, size))
            else:
                gray = cv2.cvtColor(slice_img, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (size, size))
            resized_slices.append(resized)
        
        slices = resized_slices
        
        # If we have fewer slices than needed, duplicate
        while len(slices) < noslices:
            slices.append(slices[-1] if slices else np.zeros((size, size)))
        
        # Chunk slices similar to training data
        def chunks(l, n):
            count = 0
            for i in range(0, len(l), n):
                if count < noslices:
                    yield l[i:i + n]
                    count += 1
        
        new_slices = []
        chunk_sizes = math.floor(len(slices) / noslices) if len(slices) >= noslices else 1
        if chunk_sizes == 0:
            chunk_sizes = 1
        
        for slice_chunk in chunks(slices, chunk_sizes):
            if len(slice_chunk) > 0:
                if len(slice_chunk) == 1:
                    averaged_slice = slice_chunk[0]
                else:
                    averaged_slice = np.mean(slice_chunk, axis=0)
                new_slices.append(averaged_slice)
            if len(new_slices) >= noslices:
                break
        
        # Ensure we have exactly noslices
        while len(new_slices) < noslices:
            if len(new_slices) > 0:
                new_slices.append(new_slices[-1].copy())
            else:
                new_slices.append(np.zeros((size, size)))
        
        new_slices = new_slices[:noslices]
        
        # Validate shapes
        validated_slices = []
        for s in new_slices:
            if isinstance(s, np.ndarray):
                if s.shape == (size, size):
                    validated_slices.append(s)
                else:
                    validated_slices.append(cv2.resize(s, (size, size)))
            else:
                validated_slices.append(np.zeros((size, size)))
        
        final_array = np.array(validated_slices, dtype=np.float32)
        
        if final_array.shape != (noslices, size, size):
            if len(final_array.shape) == 2:
                final_array = np.expand_dims(final_array, 0)
                while final_array.shape[0] < noslices:
                    final_array = np.vstack([final_array, final_array[0:1]])
                final_array = final_array[:noslices]
            final_array = final_array.reshape(noslices, size, size)
        
        # Transpose to (height, width, slices) for model: from (5, 10, 10) to (10, 10, 5)
        final_array = final_array.transpose(1, 2, 0)
        
        return final_array

    # Function to predict on a single image
    def predict_image(self):
        """Predict cancer on a manually selected image"""
        # Check if model exists
        if not os.path.exists('lung_cancer_model.ckpt.meta'):
            messagebox.showerror("Error", "Model not found! Please train the model first.")
            return
        
        # Open file dialog to select image
        file_path = filedialog.askopenfilename(
            title="Select Image to Predict",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.dcm *.dicom"),
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("DICOM files", "*.dcm *.dicom"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
        
        try:
            # Preprocess the image
            messagebox.showinfo("Processing", "Processing image... Please wait.")
            processed_image = self.preprocess_single_image(file_path, size=self.size, noslices=self.NoSlices)
            
            if processed_image is None:
                messagebox.showerror("Error", "Failed to process image. Please check the file format.")
                return
            
            # Load and run prediction
            x = tf.placeholder('float')
            size = 10
            keep_rate = 0.8
            NoSlices = 5
            
            def convolution3d(x, W):
                return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')
            
            def maxpooling3d(x):
                return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')
            
            def cnn(x):
                x = tf.reshape(x, shape=[-1, size, size, NoSlices, 1])
                convolution1 = tf.nn.relu(
                    convolution3d(x, tf.Variable(tf.random_normal([3, 3, 3, 1, 32]))) + tf.Variable(tf.random_normal([32])))
                convolution1 = maxpooling3d(convolution1)
                convolution2 = tf.nn.relu(
                    convolution3d(convolution1, tf.Variable(tf.random_normal([3, 3, 3, 32, 64]))) + tf.Variable(tf.random_normal([64])))
                convolution2 = maxpooling3d(convolution2)
                convolution3 = tf.nn.relu(
                    convolution3d(convolution2, tf.Variable(tf.random_normal([3, 3, 3, 64, 128]))) + tf.Variable(tf.random_normal([128])))
                convolution3 = maxpooling3d(convolution3)
                convolution4 = tf.nn.relu(
                    convolution3d(convolution3, tf.Variable(tf.random_normal([3, 3, 3, 128, 256]))) + tf.Variable(tf.random_normal([256])))
                convolution4 = maxpooling3d(convolution4)
                convolution5 = tf.nn.relu(
                    convolution3d(convolution4, tf.Variable(tf.random_normal([3, 3, 3, 256, 512]))) + tf.Variable(tf.random_normal([512])))
                convolution5 = maxpooling3d(convolution5)
                flattened = tf.keras.layers.Flatten()(convolution5)
                fullyconnected = tf.keras.layers.Dense(256, activation=tf.nn.relu)(flattened)
                fullyconnected = tf.nn.dropout(fullyconnected, keep_prob=keep_rate)
                output = tf.keras.layers.Dense(2)(fullyconnected)
                return output
            
            prediction = cnn(x)
            saver = tf.train.Saver()
            
            with tf.Session() as session:
                # Restore the trained model
                saver.restore(session, 'lung_cancer_model.ckpt')
                print('Model restored for prediction')
                
                # Make prediction
                pred_result = session.run(tf.nn.softmax(prediction), feed_dict={x: [processed_image]})
                predicted_class = np.argmax(pred_result[0])
                confidence = pred_result[0][predicted_class] * 100
                
                # Display result
                if predicted_class == 1:
                    result_text = f"CANCER DETECTED\nConfidence: {confidence:.2f}%"
                    messagebox.showwarning("Prediction Result", result_text)
                else:
                    result_text = f"NO CANCER DETECTED\nConfidence: {confidence:.2f}%"
                    messagebox.showinfo("Prediction Result", result_text)
                
                print(f"Prediction: {'Cancer' if predicted_class == 1 else 'No Cancer'}, Confidence: {confidence:.2f}%")
        
        except Exception as e:
            messagebox.showerror("Error", f"Error during prediction: {str(e)}")
            import traceback
            traceback.print_exc()

# For GUI
if __name__ == "__main__":
        root=Tk()
        obj=LCD_CNN(root)
        root.mainloop()