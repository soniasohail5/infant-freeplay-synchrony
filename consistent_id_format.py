import os 
import json

# Directory
folder_dir = "/mnt/c/3HYPER FREEPLAY DV METRABS/Processed Data/Test/3HYPER.0"

# Enter Subject ID 
subject_num = int(input("Enter Subject ID: "))

# Create full folder path (frame)
folder_path = folder_dir + str(subject_num) + " FREEPLAY DV EXTRACTED"

if not os.path.exists(folder_path):
    print("Subject ID not found.")
else:
    print("Opening folder: " + folder_path)
    
    for frame in sorted(os.listdir(folder_path)):
        if frame.endswith(".json"):
            open_file = open(os.path.join(folder_path, frame), 'r')
            print("Opening .... " + frame)
            dyad_info = json.load(open_file)
            open_file.close()
        else:
            continue
        
        for person in dyad_info["people"]:
            if type(person["person_id"]) is list:
                person["person_id"] = person["person_id"][0] # changes IDs from lists to ints for easier manipulation and consistency
                print(type(person["person_id"])) 
                
                
        # Save the updated dyad_info dictionary into the same file path
        updated_file_path = os.path.join(folder_path, frame)
        
        with open(updated_file_path, 'w') as new_file:
            json.dump(dyad_info, new_file)

            new_file.close()
            dyad_info.clear()

        
                
                
    