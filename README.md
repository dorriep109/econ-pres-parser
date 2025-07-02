# Directory layout:

 ocr-work - holds the scripts used for extracting texts from the archive pdfs using ocr. The software used is tesseract, and I trained a specific model for the 1971 file that is included in the folder.\n
 archives - holds the full text file, as accurate as I could get it, for each of the years. This consists of 1950, 1960, 1971, and 1980. These were either accurate enough on archive.org to be pulled straight from the website, or I scraped them myself using tesseract. Depends on the clarity/resolutation of scan for each year.\n
 parser.py - the main proccess for pulling name and college data from the txt files. Outputs the results into a csv file in the results folder.\n
 results - holds the csv results for each year. For each csv, if a row has a blank cell it means that there is likely a college there, but due to spelling errors the code could not pick it up.
 
