import withkeras as wk

CNN = wk.CNN_image_classifier()
option = input('=========Run sample classifier trainer with keras=========\nAuthor:William Akuffo\nElectrical and Electronic Engineering\n1.predict\n2.train classifier\nOption>> ')
if int(option) == 1:
    CNN.selectWithFileBrowser()
if int(option) == 2:
    CNN.train('training_set','test_set')


    #CNN.train('training_set','test_set')
#CNN = CNN_image_classifier()
#CNN.predict('test_set/dogs/dog.4014.jpg')
#CNN.predict('test_set/cats/cat.4007.jpg')
#CNN.selectWithFileBrowser()
#CNN.visualize()
