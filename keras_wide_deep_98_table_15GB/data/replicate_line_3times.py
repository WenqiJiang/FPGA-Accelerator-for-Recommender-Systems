output_str = ""

fname = './pred/pred1'

with open (fname,'r') as f:
    lines = f.readlines()
    print(len(lines))
    for line in lines:
        line = line.replace("\n", "")
        elementes = line.split("\t")
        if fname == './pred/pred1':
            new_label_feature = elementes * 3
        else:
            label = elementes[0]
            feature = elementes[1:]
            new_label_feature = [label] + feature * 3
        new_line = '\t'.join(new_label_feature) + '\n'
        output_str += new_line

with open (fname, 'w+') as f:
    f.write(output_str)