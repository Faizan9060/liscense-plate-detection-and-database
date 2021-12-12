import mysql.connector


#function to convert binary image to proper format
def write_file(data,filename):
	with open(filename,'wb') as file:
		file.write(data)

#function to read blob data
def readblob(id,image_path,cursor):
	
	
	query = "SELECT * FROM numberplate WHERE id=%s"
	values = (id,)
	cursor.execute(query,values)
	record = cursor.fetchall()
	for row in record:
		binary_image = row[0]
		print("extracted_text= ",row[1])
		print("date = ",row[2])
		print("time = ",row[3])
		print("id = ",row[4])
		
		write_file(binary_image,image_path)

connection = mysql.connector.connect(host='localhost',user='root',password='password',database='lp_detection')

cursor=connection.cursor()

count_query = "SELECT * FROM numberplate"
cursor.execute(count_query)
count = cursor.fetchall()

for key in range(0,len(count)+1):
	image_name = "plate_" + str(key) + ".jpg"
	image_path = "/home/faizan/Documents/license_plate_detection/retrieved_images/" + image_name
	readblob(key,image_path,cursor)
	

