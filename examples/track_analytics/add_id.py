import csv

filenames = ['exactEarth_historical_data_2017-08-10',
             'exactEarth_historical_data_2017-08-11',
             'exactEarth_historical_data_2017-08-12',
             'exactEarth_historical_data_2017-08-13',
             'exactEarth_historical_data_2017-08-14',
             'exactEarth_historical_data_2017-08-15',
             'exactEarth_historical_data_2017-08-16',
             'exactEarth_historical_data_2017-08-17',
             'exactEarth_historical_data_2017-08-18',
             'exactEarth_historical_data_2017-08-19',
             'exactEarth_historical_data_2017-08-20',
             'exactEarth_historical_data_2017-08-21',
             'exactEarth_historical_data_2017-08-22',
             'exactEarth_historical_data_2017-08-23',
             'exactEarth_historical_data_2017-08-24',
             'exactEarth_historical_data_2017-08-25',
             'exactEarth_historical_data_2017-08-26',
             'exactEarth_historical_data_2017-08-27',
             'exactEarth_historical_data_2017-08-28',
             'exactEarth_historical_data_2017-08-29',
             'exactEarth_historical_data_2017-08-30',
             'exactEarth_historical_data_2017-08-31',
             'exactEarth_historical_data_2017-09-01',
             'exactEarth_historical_data_2017-09-02',
             'exactEarth_historical_data_2017-09-03',
             'exactEarth_historical_data_2017-09-04',
             'exactEarth_historical_data_2017-09-05',
             'exactEarth_historical_data_2017-09-06',
             'exactEarth_historical_data_2017-09-07',
             'exactEarth_historical_data_2017-09-08',
             'exactEarth_historical_data_2017-09-09',
             'exactEarth_historical_data_2017-09-10']

i = 0
for filename in filenames:
    with open('data/exact_earth/{}.csv'.format(filename), 'r') as csvinput:
        print(filename)
        with open('data/exact_earth/id/{}_id.csv'.format(filename),
                  'w') as csvoutput:
            writer = csv.writer(csvoutput, lineterminator='\n')
            reader = csv.reader(csvinput)
            all = []
            row = next(reader)
            row.insert(0, 'ID')
            all.append(row)
            for row in reader:
                row.insert(0, i)
                if row[29].find("E+") > 0 and row[29][-2:] == '.0':
                    row[29] = row[29][:-2]
                if row[30].find("E+") > 0 and row[30][-2:] == '.0':
                    row[30] = row[30][:-2]
                # row[30] = row[30].replace("E+1.0", "E+1")
                all.append(row)
                i = i + 1
                # print(i)

            writer.writerows(all)
