def convertToUnix(source, destination):
    """
    convert dos linefeeds (crlf) to unix (lf)
    usage: dos2unix.py 
    """

    content = ''
    outsize = 0
    with open(source, 'rb') as infile:
        content = infile.read()
    with open(destination, 'wb') as output:
        for line in content.splitlines():
            outsize += len(line) + 1
            output.write(line + str.encode('\n'))

    print("Done. Saved %s bytes." % (len(content) - outsize))


convertToUnix("MIT_split/train_labels.dat", "MIT_split/train_labels_unix.dat")
