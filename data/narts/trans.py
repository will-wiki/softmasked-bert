

def trans(path, output):

    out = open(output, 'w')
    out.write('origin_text,random_text,label' + '\n')

    with open(path, 'r')as file:
        for line in file.readlines():
           line = line.strip().split('\t')
           origin_text = line[2]
           random_text = line[1]
           label = ' '.join(['0' if o == r else '1' for o,r in zip(origin_text, random_text)])
           input_text = origin_text + ',' + random_text + ',' + label
           out.write(input_text + '\n')
    out.close()

trans('SIGHAN15_test.txt', 'SIGHAN15_test.csv')
