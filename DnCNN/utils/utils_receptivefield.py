import math

def outFromIn(conv, layerIn):
    n_in = layerIn[0]
    j_in = layerIn[1]
    r_in = layerIn[2]
    start_in = layerIn[3]
    k = conv[0]
    s = conv[1]
    p = conv[2]
  
    n_out = math.floor((n_in - k + 2*p)/s) + 1
    actualP = (n_out-1)*s - n_in + k 
    pR = math.ceil(actualP/2)
    pL = math.floor(actualP/2)
  
    j_out = j_in * s
    r_out = r_in + (k - 1)*j_in
    start_out = start_in + ((k-1)/2 - pL)*j_in
    return n_out, j_out, r_out, start_out

def printLayer(layer, layer_name):
    print(layer_name + ":")
    print(" n features: %s  jump: %s  receptive size: %s  start: %s " % (layer[0], layer[1], layer[2], layer[3]))



layerInfos = []
if __name__ == '__main__':

    convnet =   [[3,1,1],[3,1,1],[3,1,1],[4,2,1],[2,2,0],[3,1,1]]
    layer_names = ['conv1','conv2','conv3','conv4','conv5','conv6','conv7','conv8','conv9','conv10','conv11','conv12']
    imsize = 128

    print ("-------Net summary------")
    currentLayer = [imsize, 1, 1, 0.5]
    printLayer(currentLayer, "input image")
    for i in range(len(convnet)):
        currentLayer = outFromIn(convnet[i], currentLayer)
        layerInfos.append(currentLayer)
        printLayer(currentLayer, layer_names[i])


# run utils/utils_receptivefield.py
    
