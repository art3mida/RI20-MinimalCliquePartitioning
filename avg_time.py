file_names = ['results_simple.txt', 'results_tabu.txt']
for file_name in file_names:
    with open('results/' + file_name, 'r') as input:
        with open('tables/' + file_name, 'w') as output:
            # TODO: Promeni ovo.
            for i in range(6):
                graph_name = input.readline()
                times_line = input.readline().strip('[]\n')
                times = list(map(float, times_line.split(', ')))
                worst_time = max(times)
                best_time = min(times)
                avg_time = sum(times)/len(times)


                output.write(graph_name)
                output.write('best time: {}\n'.format(best_time))
                output.write('worst time: {}\n'.format(worst_time))
                output.write('avg time: {}\n'.format(avg_time))
