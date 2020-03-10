import json
import operator

# prompt the user for a file to import
filename = "r630_deep_inference_profiling_batch_32.json"


class Measure():

    def __init__(self, name, type="layer"):
        """
        name: layer / op name
        type: "layer" or "op"
        """
        self.name = name
        self.type = type
        self.time_consumption = 0
        # stat of when each layer starts and end -> then calculate the duration
        self.time_start_list = []
        self.time_end_list = []
        self.time_start = None
        self.time_end = None
        # from start to end, may have other operators within it
        self.time_duration = None
        self.operations = dict()

class MeasureList():

    def __init__(self):
        pass

if __name__ == "__main__":

    time_consumption = dict()
    # (name, op / or layer) -> using different ways to extract duration
    time_consumption_type = [("my_model/dense_features/", "layer"), ("my_model/dense/", "layer"),
                             ("my_model/dense_1/", "layer"), ("my_model/dense_2/", "layer"),
                             ("my_model/dense_3/", "layer")]
    # time_consumption_type = [("MatMul", "op")]

    for t in time_consumption_type:
        time_consumption[t] = dict()
        # add up the total operation time of each layer
        time_consumption[t]["time_consumption"] = 0
        # stat of when each layer starts and end -> then calculate the duration
        time_consumption[t]["time_start_list"] = []
        time_consumption[t]["time_end_list"] = []
        time_consumption[t]["time_start"] = None
        time_consumption[t]["time_end"] = None
        # from start to end, may have other operators within it
        time_consumption[t]["time_duration"] = None

    operations = dict()

    # Read JSON data into the datastore variable
    with open(filename, 'r') as f:
        file = json.load(f)
        traceEvents = file["traceEvents"]

        for event in traceEvents:
            if "cat" in event and event["cat"] == "Op" and event['name'] != "unknown":

                for key in time_consumption_type:
                    if (key[1] == "op" and key[0] == event["name"]) or \
                        (key[1] == "layer" and event["args"]["name"][:len(key[0])] == key[0]):
                        time_consumption[key]["time_consumption"] += event["dur"]
                        time_consumption[key]["time_start_list"].append(event["ts"])
                        time_consumption[key]["time_end_list"].append(event["ts"] + event["dur"])

                        if event['name'] not in operations:
                            operations[event['name']] = event['dur']
                        else:
                            operations[event['name']] += event['dur']
                        print(event)

                    # if key[1] == "layer" and event["args"]["name"][:len(key[0])] == key[0]:
                    #     time_consumption[key]["time_consumption"] += event["dur"]
                    #     time_consumption[key]["time_start_list"].append(event["ts"])
                    #     time_consumption[key]["time_end_list"].append(event["ts"] + event["dur"])
                    #     print(event)

    # compute duration
    for t in time_consumption:
        time_consumption[t]["time_start_list"].sort()
        time_consumption[t]["time_end_list"].sort()
        time_consumption[t]["time_start"] = time_consumption[t]["time_start_list"][0]
        time_consumption[t]["time_end"] = time_consumption[t]["time_end_list"][-1]
        time_consumption[t]["time_duration"] = time_consumption[t]["time_end"] - time_consumption[t]["time_start"]

    # print result
    result_str = []
    for t in time_consumption:
        result = "Type: {}\tName: {}\tTime: {}\tDuration: {} (start: {}, end:{})".format(
            t[1], t[0], time_consumption[t]["time_consumption"], time_consumption[t]["time_duration"],
            time_consumption[t]["time_start"], time_consumption[t]["time_end"])
        result_str.append(result)
        print(result)

    operation_str = []
    for o in sorted(operations.keys()):
        op_str = "Operation: {}\tTime:{}".format(o, operations[o])
        operation_str.append(op_str)
        print(op_str)

    operation_str_by_time = []
    sorted_op = sorted(operations.items(), key=operator.itemgetter(1), reverse=True)
    for o in sorted_op:
        op_str = "Operation: {}\tTime:{}".format(o, operations[o[0]])
        operation_str_by_time.append(op_str)
        print(op_str)

    with open("report_" + filename[:-5], 'w+') as f:
        f.write("Measured Layer / Operations: \n")
        for item in time_consumption_type:
            f.write("\t" + str(item) + "\n")

        f.write("\n{}\n\n".format("-" * 80))
        for result in result_str:
            f.write(result + "\n")

        f.write("\n{}\n\n".format("-" * 80))
        for result in operation_str:
            f.write(result + "\n")

        f.write("\n{}\n\n".format("-" * 80))
        for result in operation_str_by_time:
            f.write(result + "\n")

#Use the new datastore datastructure
# print(datastore["office"]["parking"]["style"])