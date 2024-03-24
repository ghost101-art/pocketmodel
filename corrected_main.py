import tensorflow as tf
from PIL import Image
import json
import numpy as np
import os

# Assuming analyze_image is correctly implemented in main.py
from main import analyze_image

print(tf.__version__)


def load_tf_model(model_path):
    print("Loading TensorFlow model from:", model_path)
    tf_graph = tf.Graph()
    with tf_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    print("TensorFlow model loaded successfully.")
    return tf_graph


def preprocess_image(image_path, target_size=(224, 224)):
    print("Preprocessing image:", image_path)
    image = Image.open(image_path)
    image_resized = image.resize(target_size)
    image_np = np.array(image_resized)
    image_np_expanded = np.expand_dims(image_np, axis=0)
    return image_np_expanded


def run_inference(tf_graph, image_np_expanded):
    with tf_graph.as_default():
        with tf.compat.v1.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes',
                        'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image_np_expanded})

            # All outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            return output_dict

def predict_and_generate_results(tf_graph, image, res_file, anno_file):
    print("Loading annotations from:", anno_file)
    with open(anno_file, 'r') as file:
        annotations = json.load(file)

    img_ids = [ann['id'] for ann in annotations['images']]
    print("Loaded image IDs:", img_ids[:10])

    with tf.compat.v1.Session(graph=tf_graph) as sess:
        print("Running inference on the TensorFlow model.")
        input_tensor = tf_graph.get_tensor_by_name('image_tensor:0')
        boxes_tensor = tf_graph.get_tensor_by_name('detection_boxes:0')
        scores_tensor = tf_graph.get_tensor_by_name('detection_scores:0')
        classes_tensor = tf_graph.get_tensor_by_name('detection_classes:0')

        try:
            # Run the model
            boxes, scores, classes = sess.run(
                [boxes_tensor, scores_tensor, classes_tensor],
                feed_dict={input_tensor: image}
            )

            # Print the shapes of the outputs
            print("Shapes of the outputs:")
            print("Boxes shape:", boxes.shape)
            print("Scores shape:", scores.shape)
            print("Classes shape:", classes.shape)

            # Print the highest score to see if the model is detecting anything
            print("Highest score:", np.max(scores))

            # Optionally, print a few scores and classes for debugging
            print("Sample scores:", scores[0, :5])
            print("Sample classes:", classes[0, :5])

            results = []
            for i in range(scores.shape[1]):
                if scores[0, i] > 0.5:  # Adjust this threshold as necessary
                    results.append({
                        "image_id": img_ids[0],
                        "category_id": int(classes[0, i]),
                        "bbox": boxes[0, i].tolist(),
                        "score": scores[0, i].item(),
                    })

            if results:
                with open(res_file, 'w') as file:
                    json.dump(results, file)
                print("Prediction results saved to:", res_file)
                return res_file
            else:
                print("No detections with high confidence.")
                return None

        except Exception as e:
            print("An error occurred during inference:", e)
            return None

# Define paths
image_path = 'image.jpg'
annotations_path = 'data/sample.json'
results_file_path = 'results.json'

# Preprocess the image
preprocessed_image = preprocess_image(image_path)

tf_model_path = 'frozen_inference_graph.pb'
tf_graph = load_tf_model(tf_model_path)
# Generate predictions and results file
results_file = predict_and_generate_results(tf_graph, preprocessed_image, results_file_path, annotations_path)

# Analyze the results using your analyze_image function
if results_file:
    summary = analyze_image(anno_file=annotations_path, res_file=results_file)
    if summary:
        print("Evaluation Summary:")
        print(summary)
    else:
        print("No evaluation summary was generated, or the summary was empty.")
else:
    print("Prediction failed or no significant results were found.")
