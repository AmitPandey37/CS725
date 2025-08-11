import numpy as np

def Convolution_1D(array1: np.array, array2: np.array, padding: str, stride: int = 1) -> np.array:
    L_f = len(array1)
    L_g = len(array2)
    
    if padding == 'full':

        pad_width = L_g - 1
        array1_padded = np.pad(array1, (pad_width, pad_width), mode='constant')
    elif padding == 'valid':

        array1_padded = array1
    elif padding == 'same':

        pad_width = (L_g - 1) // 2
        array1_padded = np.pad(array1, (pad_width, pad_width), mode='constant')
    else:
        raise ValueError("Invalid padding type. Use 'full', 'valid', or 'same'.")
    

    array2_flipped = np.flip(array2)
    

    output_size = (len(array1_padded) - L_g) // stride + 1
    result = np.zeros(output_size)


    for i in range(0, output_size * stride, stride):

        result[i // stride] = np.dot(array1_padded[i:i + L_g], array2_flipped)

    return result

def probability_sum_of_faces(p_A: np.array, p_B:np.array) -> np.array:

    sum_probabilities = Convolution_1D(p_A, p_B, padding='full', stride=1)
    
    return sum_probabilities

#def test_Convolution_1D():
#    array1 = np.array([1, 2, 3, 4, 5, 6, 7])
#    array2 = np.array([1, 0, -1])
#    
#    result_valid = Convolution_1D(array1, array2, padding='valid', stride=1)
#    print(f"\nValid Padding Output: {result_valid}")
#    act_valid = np.convolve(array1, array2, mode='valid')
#    print(f"Actual Valid Padding Output: {act_valid}\n")
#
#    result_full = Convolution_1D(array1, array2, padding='full', stride=1)
#    print(f"\nFull Padding Output: {result_full}")
#    act_full = np.convolve(array1, array2, mode='full')
#    print(f"Actual Full Padding Output: {act_full}\n")
#
#    result_same = Convolution_1D(array1, array2, padding='same', stride=1)
#    print(f"\nSame Padding Output: {result_same}")
#    act_same = np.convolve(array1, array2, mode='same')
#    print(f"Actual Same Padding Output: {act_same}\n")
#
#    p_A = np.array([0.1, 0.2, 0.3, 0.1, 0.1, 0.2])  # 6 faces on die A
#    p_B = np.array([0.3, 0.4, 0.3])  # 3 faces on die B
#
#    result = probability_sum_of_faces(p_A, p_B)
#    print(f"\nProbabilities of the sum of the faces: {result}\n")
#
#test_Convolution_1D()
