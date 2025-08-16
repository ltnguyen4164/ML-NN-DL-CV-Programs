# Long Nguyen
# 1001705873

from collections import deque
import numpy as np
import random

class Tree:
    def __init__(self, attribute=None, threshold=None, gain=None, left=None, right=None, prediction=None):
        self.attribute = attribute      # feature/attribute index
        self.threshold = threshold      # threshold value for splitting
        self.gain = gain                # value of information gain
        self.left = left                # left child node
        self.right = right              # right child node
        self.prediction = prediction    # class label for leaf nodes

# function to find the mode of x 
def distribution(x):
    labels = x[:, -1]
    vals, counts = np.unique(labels, return_counts=True)
    max_count_idx = np.argmax(counts)
    return vals[max_count_idx]
# function to calculate the entropy of examples
def entropy(exp):
    labels = exp[:, -1]
    vals, counts = np.unique(labels, return_counts=True)
    total = len(exp)

    ent = 0
    for count in counts:
        prob = count/total
        ent -= prob * np.log2(prob)
    return ent
# function to calculate the information gain
def information_gain(exp, attr, thr):
    left = exp[exp[:, attr] < thr]
    right = exp[exp[:, attr] >= thr]

    total_ent = entropy(exp)
        
    left_weight = len(left) / len(exp)
    right_weight = len(right) / len(exp)

    weighted_ent = left_weight * entropy(left) + right_weight * entropy(right)
    return total_ent - weighted_ent
# function to choose the best attribute for interger option
def choose_attr_randomized(attributes, examples):
    max_gain = best_thr = -1
    attr = random.choice(attributes)
    attr_values = examples[:, attr]
    L = np.min(attr_values)
    M = np.max(attr_values)
    for K in range(1, 51):
        thr = L + K * (M - L) / 51
        gain = information_gain(examples, attr, thr)
        if gain > max_gain:
            max_gain = gain
            best_thr = thr
    return attr, best_thr, max_gain
# function to choose the best attribute for optimized option
def choose_attr_optimized(attributes, examples):
    max_gain = best_attr = best_thr = -1
    for attr in attributes:
        attr_values = examples[:, attr]
        L = np.min(attr_values)
        M = np.max(attr_values)
        for K in range(1, 51):
            threshold = L + K * (M - L) / 51
            gain = information_gain(examples, attr, threshold)
            if gain > max_gain:
                max_gain = gain
                best_attr = attr
                best_thr = threshold
    return best_attr, best_thr, max_gain
def DTL_Top_Level(examples, pruning_thr, option):
    attributes = list(range(examples.shape[1] - 1))
    default = distribution(examples)
    return DTL(examples, attributes, default, pruning_thr, option)
def DTL(examples, attributes, default, pruning_thr, option):
    if len(examples) < pruning_thr:
        return Tree(prediction=default)
    elif len(np.unique(examples[:, -1])) == 1:
        return Tree(prediction=examples[0, -1])
    else:
        # if option is an integer other than 0, create a randomized tree
        if option == 0:
            best_attr, best_thr, max_gain = choose_attr_optimized(attributes, examples)
        else:
            best_attr, best_thr, max_gain = choose_attr_randomized(attributes, examples)
        tree = Tree(attribute=best_attr, threshold=best_thr, gain=max_gain)
        left = examples[examples[:, best_attr] < best_thr]
        right = examples[examples[:, best_attr] >= best_thr]
        dist = distribution(examples)
        tree.left = DTL(left, attributes, dist, pruning_thr, option)
        tree.right = DTL(right, attributes, dist, pruning_thr, option)
        return tree

def classify(tree, obj):
    if tree.left is None and tree.right is None:
        return tree.prediction
    elif obj[tree.attribute] < tree.threshold:
        return classify(tree.left, obj)
    else:
        return classify(tree.right, obj)
def test(trees, test_data):
    object_id = 1
    total = 0

    for obj in test_data:
        true_class = obj[-1]

        # collect predictions from all trees in the forest
        predictions = []
        for tree_id, tree in enumerate(trees, start=1):
            predicted_class = classify(tree, obj)
            predictions.append(predicted_class)

        # determine the most frequent predicted class
        predicted_counts = {cls: predictions.count(cls) for cls in set(predictions)}
        max_count = max(predicted_counts.values())
        best_classes = [cls for cls, count in predicted_counts.items() if count == max_count]

        # randomly choose a class if there is a tie
        predicted_class = random.choice(best_classes) if len(best_classes) > 1 else best_classes[0]

        # compute accuracy
        if len(best_classes) == 1:
            accuracy = 1 if predicted_class == true_class else 0
        else:
            if true_class in best_classes:
                accuracy = 1 / len(best_classes)
            else:
                accuracy = 0
        total += accuracy
        
        # output necessary information
        #print(f'ID={object_id:5d}, predicted={int(predicted_class):3d}, true={int(true_class):3d}, accuracy={accuracy:4.2f}')
        object_id += 1
    class_accuracy = total / len(test_data)
    print(f"classification accuracy={class_accuracy:6.4f}")
        
def print_tree(root, tree_id):
    if root is None:
        return
    # initialize queue with root node
    queue = deque([(root, 1)])
    while queue:
        curr_node, node_id = queue.popleft()
        feature_id = curr_node.attribute + 1 if curr_node.attribute is not None else -1
        threshold = curr_node.threshold if curr_node.threshold is not None else -1
        gain = curr_node.gain if curr_node.gain is not None else 0

        # Output information
        print(f'tree={tree_id:2d}, node={node_id:3d}, feature={feature_id:2d}, thr={threshold:6.2f}, gain={gain:.6f}')

        # queue left and right children
        if curr_node.left is not None:
            queue.append((curr_node.left, 2 * node_id))
        if curr_node.right is not None:
            queue.append((curr_node.right, 2 * node_id + 1))
def parse_input(value):
    # translation dictionary for strings
    # assuming that the max number of classes is 10
    translation_dict = {
        "zero": 0.0,
        "one": 1.0,
        "two": 2.0,
        "three": 3.0,
        "four": 4.0,
        "five": 5.0,
        "six": 6.0,
        "seven": 7.0,
        "eight": 8.0,
        "nine": 9.0,
        "ten": 10.0,
        "class0": 0.0,
        "class1": 1.0,
        "class2": 2.0,
        "class3": 3.0,
        "class4": 4.0,
        "class5": 5.0,
        "class6": 6.0,
        "class7": 7.0,
        "class8": 8.0,
        "class9": 9.0,
        "class10": 10.0
        }
    
    try:
        return float(value)
    except ValueError:
        return translation_dict.get(value, value)
def decision_tree(training_file, test_file, option, pruning_thr):
    training_data = []
    test_data = []

    # try to open training file and extract data
    try:
        with open(training_file) as file:
            for line in file:
                training_list = [parse_input(value) for value in line.split()]
                training_data.append(training_list)
    except FileNotFoundError:
        print(f"Error: The file '{training_file}' does not exist.")
        quit()
    except Exception as e:
        print(f"An error occurred: {e}")
        quit()
        
    training_data = np.array(training_data)

    # try to open testing file and extract data
    try:
        with open(test_file) as file:
            for line in file:
                test_list = [parse_input(value) for value in line.split()]
                test_data.append(test_list)
    except FileNotFoundError:
        print(f"Error: The file '{test_file}' does not exist.")
        quit()
    except Exception as e:
        print(f"An error occurred: {e}")
        quit()
        
    test_data = np.array(test_data)

    # create tree based on option, and start the DTL process
    if option == "optimized":
        tree = DTL_Top_Level(training_data, pruning_thr, 0)
        print_tree(tree, 1)
        # using the tree, classify the test data
        test([tree], test_data)
    elif isinstance(option, int):
        forest = []
        for _ in range(option):
            tree = DTL_Top_Level(training_data, pruning_thr, option)
            forest.append(tree)
        for idx, tr in enumerate(forest):
            print_tree(tr, idx + 1)
        test(forest, test_data)