import subprocess
import sys
import xml.etree.ElementTree as ET
from math import sin

try:
    import pandas as pd
except ModuleNotFoundError:
    subprocess.call([sys.executable, "-m", "pip", "install", 'pandas'])
finally:
    import pandas as pd


def xml_parse(filename):
    tree = ET.parse(filename)
    root = tree.getroot()
    root_len = int(len(root))

    id, x_1, x_2, x_3, x_4, x_5, y  =  ([] for i in range(7))

    for i in range(root_len-1,-1,-1):
        id.append(root[i].get('ID'))
        x_1.append(root[i][0][0].get('VALUE'))
        x_2.append(root[i][0][1].get('VALUE'))
        x_3.append(root[i][0][2].get('VALUE'))
        x_4.append(root[i][0][3].get('VALUE'))
        x_5.append(root[i][0][4].get('VALUE'))
        y.append(round(0.5*float(root[i][0][0].get('VALUE')) + float(root[i][0][1].get('VALUE')) + 0.5 * float(root[i][0][0].get('VALUE'))
                * float(root[i][0][1].get('VALUE')) + 5 * sin(float(root[i][0][2].get('VALUE'))) +
                       0.2 * float(root[i][0][3].get('VALUE')) + 0.1 * float(root[i][0][4].get('VALUE')),2))


    return id, x_1, x_2, x_3, x_4, x_5, y


def main():
    filename = 'inputs.xml'
    id, x_1, x_2, x_3, x_4, x_5, y = xml_parse(filename)

    df = pd.DataFrame({'ID': id, 'X1': x_1, 'X2': x_2, 'X3': x_3, 'X4': x_4, 'X5': x_5, 'Y': y})
    df.to_csv('xml_out.csv', index=False)
    #print(df.to_string(index=False)) ## Uncomment to view the tabular format


if __name__ == "__main__":
    main()


