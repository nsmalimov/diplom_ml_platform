import urllib2
s = urllib2.urlopen("http://127.0.0.1:5000/project_load_all").read()

print (s)