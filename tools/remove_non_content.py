import sys

f_in = sys.argv[1]
f_out = sys.argv[2]
count = 0
with open(f_out, 'w') as f:
    for l in open(f_in):
        spl = l.strip().split('\t')
        if len(spl) < 4:
            f.write(l)
            count += 1
        else:
            # content only
            #f.write(f'{spl[0]}\t{spl[3]}\n')
            # title+body
            f.write(f'{spl[0]}\t{spl[2]} {spl[3]}\n')
            # title
            #f.write(f'{spl[0]}\t{spl[2]}\n')

   
print(count)
