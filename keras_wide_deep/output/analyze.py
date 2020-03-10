import json
import operator

# prompt the user for a file to import
filename = "r630_deep_inference_profiling_batch_1024.json"


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

    def compute_duration(self):
        self.time_start_list.sort()
        self.time_end_list.sort()
        self.time_start = self.time_start_list[0]
        self.time_end = self.time_end_list[-1]
        self.time_duration = self.time_end - self.time_start

    def get_time_consumption(self):
        """
        return a string that illustrate the time consumption
        """
        time_str = "Type: {}\tName: {}\tTime: {}\tDuration: {} (start: {}, end:{})\n".format(
            self.type, self.name, self.time_consumption, self.time_duration, self.time_start, self.time_end)

        return time_str

    def get_operation_time(self):
        """
        return a string that illustrate the time consumption
        """
        operation_str = ""
        sorted_op = sorted(self.operations.items(), key=operator.itemgetter(1), reverse=True)
        for i, o in enumerate(sorted_op):
            operation_str += "{}\tOperation: {}\tTime:{}\n".format(i + 1, o, self.operations[o[0]])

        return operation_str

class MeasureList():

    def __init__(self, measure_list):
        """
        measure_list: a list of Measure Objects
        """
        self.measure_list = measure_list
        self.op_list = [item.operations for item in measure_list]
        self.operations = dict()
        self.operation_statistics()

    def operation_statistics(self):
        """
        add up operations consumption of all layers
        """
        for item in self.op_list:
            for key in item:
                if key not in self.operations:
                    self.operations[key] = item[key]
                else:
                    self.operations[key] += item[key]

    def get_operation_time(self):
        """
        return a string that illustrate the time consumption
        """
        operation_str = ""
        sorted_op = sorted(self.operations.items(), key=operator.itemgetter(1), reverse=True)
        for i, o in enumerate(sorted_op):
            operation_str += "{}\tOperation: {}\tTime:{}\n".format(i + 1, o, self.operations[o[0]])

        return operation_str


if __name__ == "__main__":

    time_consumption = dict()
    # (name, op / or layer) -> using different ways to extract duration
    time_consumption_type = [("my_model/dense_features/", "layer"), ("my_model/dense/", "layer"),
                             ("my_model/dense_1/", "layer"), ("my_model/dense_2/", "layer"),
                             ("my_model/dense_3/", "layer")]
    # time_consumption_type = [("MatMul", "op")]

    measure_list = []
    for item in time_consumption_type:
        measure_list.append(Measure(item[0], item[1]))

    # Read JSON data into the datastore variable
    with open(filename, 'r') as f:
        file = json.load(f)
        traceEvents = file["traceEvents"]

        for event in traceEvents:
            if "cat" in event and event["cat"] == "Op" and event['name'] != "unknown":

                for item in measure_list:
                    if (item.type == "op" and item.name == event["name"]) or \
                        (item.type == "layer" and event["args"]["name"][:len(item.name)] == item.name):

                        item.time_consumption += event["dur"]
                        item.time_start_list.append(event["ts"])
                        item.time_end_list.append(event["ts"] + event["dur"])

                        if event['name'] not in item.operations:
                            item.operations[event['name']] = event['dur']
                        else:
                            item.operations[event['name']] += event['dur']
                        print(event)

    # compute duration
    for item in measure_list:
        item.compute_duration()

    op_measure_list = MeasureList(measure_list)

    with open("report_" + filename[:-5], 'w+') as f:

        f.write("Measured Layer / Operations: \n")
        for item in time_consumption_type:
            f.write("\t" + str(item) + "\n")

        f.write("\n{}\n\n".format("-" * 80))
        for item in measure_list:
            f.write(item.get_time_consumption())

        for item in measure_list:
            f.write("\n{}\n\n".format("-" * 80))
            f.write("Layer: {}\n\n".format(item.name))
            f.write(item.get_operation_time())

        f.write("\n{}\n\n".format("-" * 80))
        f.write("All layer:\n\n")
        f.write(op_measure_list.get_operation_time())
