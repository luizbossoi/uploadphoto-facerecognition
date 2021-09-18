import os, sys
import cv2
import uuid
import numpy as np
from scipy import spatial
from face_extraction import FaceExtraction
from face_embedding import FaceEmbedding
from flask import Flask, flash, request, redirect, url_for, jsonify, render_template
from werkzeug.utils import secure_filename

MIN_SIM             = 78
UPLOAD_FOLDER       = 'uploads/'
ALLOWED_EXTENSIONS  = set(['jpg', 'jpeg'])
modelPath           = "face_detection_model/"
embeddingModel      = "face_embedding_model/openface_nn4.small2.v1.t7"
out_dir             = "output/"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "lk3jgKAJH3OGJa2io13iy1i3iMSKAK3W91932JRkscKLSA3F3"

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload_file(fileid):
    retfiles    = []
    counter     = 0
    
    if request.method == 'POST':
        files = request.files.getlist("file")

        for file in files:
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
                
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                extension = os.path.splitext(filename)[1][1:]
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], fileid + "_" + str(counter) + "." + extension))
                retfiles.append(fileid + "_" + str(counter) + "." + extension)
            else:
                return False
           
            counter = counter + 1
                
        return retfiles

def face_matching(id_path, selfie_path, fileid):
    same_person     = False  
    found_face_id   = False
    counter         = 0
    
    while (found_face_id == False):
        idImage = cv2.imread(UPLOAD_FOLDER + id_path)
        detect  = FaceExtraction(idImage, modelPath)

        counter = counter + 1
        faces   = detect.detect_face()
    
        if len(faces)==0:
            print("No face detected on ID")
            
            if(counter>=4):
                return jsonify(error=True, where="face_id_not_found", message="no face found on ID", tries=counter)
            else:
                print("Rotate image")
                img_rotate_90_clockwise = cv2.rotate(idImage, cv2.cv2.ROTATE_90_CLOCKWISE)
                cv2.imwrite(UPLOAD_FOLDER + id_path, img_rotate_90_clockwise)
        else:
            found_face_id = True
    
    if len(faces) > 1:
        print("More than 1 faces detected in the ID image\nPlease provide another ID!!!")
        return jsonify(error=True, where="face_id_more_than_one", message="more than one face on ID")
    else:
        cv2.imwrite(out_dir + fileid + '_A001.png', faces[0])
        faceEmbeddingVec = FaceEmbedding(faces[0], embeddingModel)
        embeddingVectorId = faceEmbeddingVec.get_face_embedding()
    

    found_face_selfie = False
    counter = 0
    while (found_face_selfie == False):
        counter         = counter + 1
        selfieImage     = cv2.imread(UPLOAD_FOLDER + selfie_path)
        detect          = FaceExtraction(selfieImage, modelPath)
        faces           = detect.detect_face()
        
        if(len(faces)==0):
            print("No face detected on Selfie")
            
            if(counter>=4):
                return jsonify(error=True, where="face_selfie_not_found", message="no face found on selfie", tries=counter)
            else:
                print("Rotate image")
                img_rotate_90_clockwise = cv2.rotate(selfieImage, cv2.cv2.ROTATE_90_CLOCKWISE)
                cv2.imwrite(UPLOAD_FOLDER + selfie_path, img_rotate_90_clockwise)
        else:
            found_face_selfie = True
        
    if len(faces) > 1:
        print("More than 1 faces detected in the Selfie\nPlease provide another Selfie!!!")
        return jsonify(error=True, where="face_selfie_more_than_one", message="more than one face on selfie photo")
    else:
        cv2.imwrite(out_dir + fileid + '_B001.png', faces[0])
        faceEmbeddingVec = FaceEmbedding(faces[0], embeddingModel)
        embeddingVectorSelfie = faceEmbeddingVec.get_face_embedding()

        similarity_dist = round((spatial.distance.cosine(embeddingVectorId, embeddingVectorSelfie) -1)*-1,2)
        
        if(similarity_dist==1): similarity_dist = 100
        
        if(similarity_dist > MIN_SIM/100):
            same_person = True

        return jsonify(error=False, same_person=same_person, similarity="{}%".format(similarity_dist).replace("0.",""))


@app.route("/")
def main():
    return jsonify(result=False, message="Error, missing request")

@app.route("/front")
def front():
    return render_template('front.html')

@app.route("/front_result")
def front_result():
    same_person = request.args.get('same_person')
    similarity = request.args.get('similarity')
    person_result = "<h4><span style=\"color:green;font-weight:bold\">São a mesma pessoa</span></h4>" if (same_person=='True') else "<h4><span style=\"color:red;font-weight:bold\">Não são a mesma pessoa</span></h4>"
    return render_template('front_result.html', person_result=person_result, similarity=similarity)

@app.route("/front_error")
def front_error():
    error = request.args.get('error')
    return render_template('front_error.html', error=error)


@app.route("/recognize", methods=['GET','POST'])
def recognize():
    fileid      = str(uuid.uuid1())

    if request.method != 'POST':
        return jsonify(result=False, message="Error, method not allowed")
    else:
        print("Upload Requested")
        photos = upload_file(fileid)

        if(photos==False):
            return jsonify(error=True, message="Image not found Error, image types not allowed, only JPG/JPEG allowed")
            
        print("File Uploaded")
        print("Requesting analysis...")
        ret = face_matching(photos[0], photos[1], fileid)

        # remove all uploaded photos
        for photo in photos:
            os.remove('./' + UPLOAD_FOLDER + photo)
            
        # remove all output
        if(os.path.exists('./' + out_dir + fileid + '_A001.png')): os.remove('./' + out_dir + fileid + '_A001.png')
        if(os.path.exists('./' + out_dir + fileid + '_B001.png')): os.remove('./' + out_dir + fileid + '_B001.png')
        
        
        if(request.args.get('front')):
            ret = ret.get_json()
            if(ret['error']==False):
                return redirect("/front_result?same_person=" + str(ret['same_person']) + '&similarity=' + str(ret['similarity']), code=302)
            else:
                return redirect("/front_error?error=" + str(ret['message']), code=302)
            
        return ret


app.run(host='0.0.0.0', port=3000)