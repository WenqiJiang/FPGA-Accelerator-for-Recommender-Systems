import json

# prompt the user for a file to import
filename = "inference_profiling.json"

#Read JSON data into the datastore variable
if filename:
    with open(filename, 'r') as f:
        file = json.load(f)
        traceEvents = file["traceEvents"]
        print(file)
        print(traceEvents) # a list of dict, each dictionary is a operation, dataflow, etc.

        time_consumption = dict()
        # (name, op / or layer) -> using different ways to extract duration
        time_consumption_type = [("MatMul", "op"), ("embedding", "layer"), ("flatten", "layer"),
                                 ("dense", "layer"), ("concatenate", "layer")]
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

        for event in traceEvents:
            if "cat" in event and event["cat"] == "Op":

                for key in time_consumption_type:
                    if key[1] == "op" and key[0] == event["name"]:
                        time_consumption[key]["time_consumption"] += event["dur"]
                        time_consumption[key]["time_start_list"].append(event["ts"])
                        time_consumption[key]["time_end_list"].append(event["ts"] + event["dur"])
                        print(event)

                    if key[1] == "layer" and event["args"]["name"][:len(key[0])] == key[0]:
                        time_consumption[key]["time_consumption"] += event["dur"]
                        time_consumption[key]["time_start_list"].append(event["ts"])
                        time_consumption[key]["time_end_list"].append(event["ts"] + event["dur"])
                        print(event)

        # compute duration
        for t in time_consumption:
            time_consumption[t]["time_start_list"].sort()
            time_consumption[t]["time_end_list"].sort()
            time_consumption[t]["time_start"] = time_consumption[t]["time_start_list"][0]
            time_consumption[t]["time_end"] = time_consumption[t]["time_end_list"][-1]
            time_consumption[t]["time_duration"] = time_consumption[t]["time_end"] - time_consumption[t]["time_start"]

        # print result
        for t in time_consumption:
            print("Type: {}\tName: {}\tTime: {}\tDuration: {}".format(
                t[1], t[0], time_consumption[t]["time_consumption"], time_consumption[t]["time_duration"]))


#Use the new datastore datastructure
# print(datastore["office"]["parking"]["style"])