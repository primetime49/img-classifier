import coremltools
import PIL.Image
import csv

def load_image(path, resize_to=None):
    # resize_to: (Width, Height)
    img = PIL.Image.open(path)
    if resize_to is not None:
        img = img.resize(resize_to, PIL.Image.ANTIALIAS)
    return img

arr = {}
arr['filename'] = []
arr['category'] = []

with open('test.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        temp = row[0].split(",")
        print(temp[0])
        arr['filename'].append(temp[0])
        if temp[0] == 'filename':
            arr['category'].append(temp[1])
            continue
        tmp = 'test/test/' + temp[0]
        # print(tmp)
        model = coremltools.models.MLModel('ImageClassifier.mlmodel')
        img = load_image(tmp, resize_to=(150, 150))
        result = model.predict({'image': img})
        # print(result)
        arr['category'].append(result['classLabel'])
        # break

with open('test.csv', mode='w') as test_file:
    test_writer = csv.writer(test_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    for i in range(len(arr['filename'])):
        test_writer.writerow([arr['filename'][i], arr['category'][i]])