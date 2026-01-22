"""
A Python script to import markers positions (in image geometry) into a Metashape project.
To be run in Metashape with "Tools -> Run Script".
Used to check photogrammetric targets locations.

Date: Jan 2026
"""

#markers batch import script
##data format:
#image_name, gcp_name, px_x, px_y
##sample data:
#DJI_0470.JPG,GCP_1,2664,3207
#

import Metashape, os

def import_markers_csv():

    doc = Metashape.app.document
    if not (len(doc.chunks)):
        print("No chunks. Script aborted.\n")
        return False

    path = Metashape.app.getOpenFileName("Select markers import file:")
    if path == "":
        print("Incorrect path. Script aborted.\n")
        return False
    print("Markers import started.\n")  #informational message
    file = open(path, "rt")	#input file
    print(path)
    eof = False
    line = file.readline() #skipping header line
    line = file.readline()
    if not len(line):
        eof = True

    chunk = doc.chunk

    while not eof:	
        sp_line = line.strip().rsplit(",", 3)   #splitting read line
        print(sp_line)
        y = float(sp_line[3])			#x- coordinate of the current projection in pixels
        x = float(sp_line[2])			#y- coordinate of the current projection in pixels
        label = sp_line[1]				#image file name
        marker_name = sp_line[0]		#marker label

        flag = 0
        for camera in chunk.cameras:

                if os.path.basename(camera.photo.path).lower() == label.lower():		#searching for the camera

                        for marker in chunk.markers:	#searching for the marker (comparing with all the marker labels in chunk)
                                if marker.label.lower() == marker_name.lower():
                                        marker.projections[camera] =  Metashape.Marker.Projection(Metashape.Vector([x,y]), True)		#setting up marker projection of the correct photo)
                                        flag = 1
                                        break

                        if not flag:   #creating new marker instance
                                marker = chunk.addMarker() 
                                marker.label = marker_name
                                marker.projections[camera] =  Metashape.Marker.Projection(Metashape.Vector([x,y]), True)

                        break

        line = file.readline()		#reading the line from input file
        if not len(line):
                eof = True
                break # EOF

    #chunk.crs = Metashape.CoordinateSystem("EPSG::4326")
    file.close()	
    print ("Markers import finished.\n")
    return True

Metashape.app.addMenuItem("Custom menu/Import markers from CSV file", import_markers_csv)	
