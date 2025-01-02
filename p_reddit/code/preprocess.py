import os
import xml.etree.ElementTree as ET
import pandas as pd

# Directory containing the XML files
path_base = "/home/marco/Desktop/reddit_depression/p_reddit/dataset"
chunks = [f"{path_base}/chunk{i}" for i in range(1, 11)]
# Write the DataFrame to a CSV file
output_xml_path = "/home/marco/Desktop/reddit_depression/p_reddit/dataset_revised/"
os.makedirs(output_xml_path, exist_ok=True)


# List to hold all records
records = []
i=1
# Iterate over all XML files in the directory
for xml_directory in chunks:
    for filename in os.listdir(xml_directory):
        if filename.endswith(".xml"):
            file_path = os.path.join(xml_directory, filename)
            tree = ET.parse(file_path)
            root = tree.getroot()

            # Extract data from XML
            for writing in root.findall('WRITING'):
                record = {
                    "ID": root.find('ID').text,
                    "TITLE": writing.find('TITLE').text,
                    "DATE": writing.find('DATE').text,
                    "TEXT": writing.find('TEXT').text
                }
                records.append(record)
    
    # Create a new XML tree
    root = ET.Element("RECORDS")
    for record in records:
        writing = ET.SubElement(root, "WRITING")
        ET.SubElement(writing, "ID").text = record["ID"]
        ET.SubElement(writing, "TITLE").text = record["TITLE"]
        ET.SubElement(writing, "DATE").text = record["DATE"]
        ET.SubElement(writing, "TEXT").text = record["TEXT"]

    tree = ET.ElementTree(root)
    tree.write(f"{output_xml_path}chunk{i}.xml", encoding="utf-8", xml_declaration=True)
    records.clear()
    i+=1

print(f"Data successfully merged and written to {output_xml_path}")