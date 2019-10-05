from core import Core


c = Core()

image_filename = c.current_path + "/DataSets/Drones/testImages/351.jpg"
image = c.load_image_by_path(image_filename)

drawing_image = c.get_drawing_image(image)

processed_image, scale = c.pre_process_image(image)

c.set_model(c.get_model())
boxes, scores, labels = c.predict_with_graph_loaded_model(processed_image, scale)

detections = c.draw_boxes_in_image(drawing_image, boxes, scores)

c.visualize(drawing_image)
