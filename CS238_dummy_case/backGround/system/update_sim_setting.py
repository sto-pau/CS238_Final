
if __name__ == '__main__':
    path = 'controlDict'

    new_start_time = 0.1 #0
    new_end_time = 200 #100
    new_writeInterval = 0.4 #0.2


    with open(path, 'r') as file:
        file_lines = file.readlines()

    file.close()

    for line_number , line_content in enumerate(file_lines):
        if line_content.startswith("startTime"):
            '''can't use this method on ssh w/ lower version python
            file_lines[line_number] = f"startTime       {new_start_time};\n" #7 spaces
            '''
        elif line_content.startswith("endTime"):
            file_lines[line_number] = f"endTime         {new_end_time};\n" #9 spaces
        elif line_content.startswith("writeInterval"):
            file_lines[line_number] = f"writeInterval   {new_writeInterval};\n" #3 spaces

    with open(path, 'w') as file:
        file.writelines(file_lines)

    file.close()