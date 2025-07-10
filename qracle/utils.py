import matplotlib.pyplot as plt



import smtplib 
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


import h5py 
import random 

from pennylane import numpy as np



# Split the data into training and validation sets 
def split_data(data_file, split_ratio=0.7):
    temp_data_file_name = data_file.split('.')[0]

    train_data_file = temp_data_file_name + '_train.h5'
    valid_data_file = temp_data_file_name + '_valid.h5'


    random.seed(42)


    with h5py.File(data_file, 'r') as f_in:
        group_names = list(f_in.keys())
        first_group_name = group_names[0]
        keys = list(f_in[first_group_name].keys())
        
        random.shuffle(keys)
        
        total_num_samples = len(keys)

        train_count = int(total_num_samples * split_ratio)


        train_keys = set(random.sample(keys, train_count))
        valid_keys = set(keys) - train_keys
        with h5py.File(train_data_file, 'w') as f_train, h5py.File(valid_data_file, 'w') as f_valid:
            for group_name in f_in.keys():
                train_group = f_train.create_group(group_name)
                valid_group = f_valid.create_group(group_name)

                for key in keys:
                    data = f_in[group_name][key][...]
                    if key in train_keys:
                        train_group.create_dataset(key, data=data)
                    elif key in valid_keys:
                        valid_group.create_dataset(key, data=data)


    return train_data_file, valid_data_file


def rename_group(h5_file, old_group_name, new_group_name):
    with h5py.File(h5_file, "r+") as f:
        if old_group_name in f:
            # Create a new group with the new name
            f.copy(old_group_name, new_group_name)
            # Delete the old group
            del f[old_group_name]
            print(f"Renamed '{old_group_name}' → '{new_group_name}'")
        else:
            print(f"Group '{old_group_name}' not found!")


def get_gnn_results(old_init, new_init):
    old_init = np.array(old_init)
    new_init = np.array(new_init)




    improved_init = old_init >= new_init

    worsened_init = old_init < new_init

    improved_cnt = np.sum(improved_init)
    worsened_cnt = np.sum(worsened_init)
    print(f"Improved initial loss: {improved_cnt} / {len(old_init)}")
    print(f"Worsened initial loss: {worsened_cnt} / {len(old_init)}")
    categories = ['lower than random init', 'higher  than random init']
    counts = [improved_cnt, worsened_cnt]  # Given data

    # Create the bar chart
    plt.figure(figsize=(8, 5))
    bars = plt.bar(categories, counts, color=['green', 'orange'])

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height, str(height),
                ha='center', va='bottom', fontsize=12, fontweight='bold')


    # Add labels and title
    plt.xlabel("Categories")
    plt.ylabel("Count")
    plt.title(f"Classification Outcomes Distribution out of {len(old_init)} circuits")

    # Show the plot
    plt.show()



def send_email(subject, body, recipient_email=' '): #Rplace your email inside ' '

    sender_email = ' '  # Replace with your email
    sender_password = ' ' # Replace with your email password


     # Create the email
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))


    try:
    # Connect to the server and send the email
        with smtplib.SMTP('smtp.mail.me.com',587) as server:  # For Gmail
            server.starttls()  # Secure the connection
            server.login(sender_email, sender_password)
            server.send_message(msg)
            print("Email sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")