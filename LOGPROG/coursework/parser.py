# "нечто", выполняющее парсинг .ged файлов в утверждения на языке Prolog
# использование parser.py <входной (.ged файл)> <выходной .pl> <[-t] ключ для перевода на транслит>
import re
import sys

trslt = False

if '-t' in sys.argv:
	trslt = True

inp = sys.argv[1]
outp = sys.argv[2]

# перевод имен в транслит если интерпретатор не поддерживает кириллицу
def translit(s):
	d = {
		'А' : 'A', 'Б' : 'B', 'В' : 'V', 'Г' : 'G', 'Д' : 'D',
		'Е' : 'E', 'Ё' : 'YO', 'Ж' : 'J', 'З' : 'Z', 'И' : 'I',
		'Й' : 'Y', 'К' : 'K', 'Л' : 'L', 'М' : 'M', 'Н' : 'N',
		'О' : 'O', 'П' : 'P', 'Р' : 'R', 'С' : 'S', 'Т' : 'T',
		'У' : 'U', 'Ф' : 'F', 'Х' : 'H', 'Ц' : 'C',	'Ч' : 'CH',
		'Ш' : 'SH', 'Щ' : 'SCH', 'Ъ' : '', 'Ы' : '', 'Ь' : '',
		'Э' : 'E', 'Ю' : 'YU', 'Я' : 'YA', ' ' : ' '
	}
	s = s.upper()
	res = ""
	for ch in s:
		res+=d[ch]
	return res.capitalize()
if len(sys.argv) < 3:
	print("Using: parser.py <input(.ged file)> <output (.pl file)> <[-t] for translit names> ")
	exit()

d = {}

persons = {}
fams = {}

infile = open("my.ged", "r")

# разбивка на записи:

# person -- словарь индивидов в формате <1> : [<2>, <3>, <4>]
# 1 -- id индивида в .ged файле
# 2 -- имя индивида
# 3 -- пол индивида (M\F)
# 4 -- если есть семья, то id семьи, иначе 0

# fams -- словарь семей в формате <1> : [<2>, <3>, <4>]
# 1 -- id семьи в .ged файле
# 2 -- имя мужа
# 3 -- имя жены
# 4 -- список потомков

for line in infile:

	words = line.strip().split()
	d = {}
	indi = None

	if "INDI" in words:
		indi = words[1][2:-1]
		while True:
			l = infile.readline().strip().split()
			if l[1] == "_UID":
				break
			d[l[1]] = l[2:]
	if len(d.keys()) :
		hasCh = d["FAMS"][0][2:-1] if "FAMS" in d.keys() else 0
		if trslt: 
			persons[indi] = [translit(" ".join(d['GIVN'])+" "+" ".join(d['SURN'])), d['SEX'][0], hasCh]
		else:
			persons[indi] = [" ".join(d['GIVN'])+" "+" ".join(d['SURN']), d['SEX'][0], hasCh]
		d.clear()	

	if "FAM" in words:
		fam = words[1][2:-1]
		d["CHIL"] = []
		while True:
			l = infile.readline().strip().split()
			if l[1] == "_UID":
				break
			if l[1] == "CHIL":
				d[l[1]].append(l[2][2:-1])
			else:
				d[l[1]] = l[2][2:-1]
	if len(d.keys()) :
		fams[fam] = [d["HUSB"], d["WIFE"], d["CHIL"]]
		d.clear()
infile.close()

output = []

# В каждой семье для каждого ребенка формируются факты родства в формате
# [<мать\отец>(<мать\отец>,<ребенок>).], в зависимости от пола родителя.

# Если у ребенок в семье еще нет своих детей, то для него фомируется факт 
# [<мать\отец>(<ребенок>, nobody)], предполагая, что в будующем, весьма 
# вероятно, они у него появятся, а пол останется тот же.

# Факты данного вида вводятся как решение проблемы невозможности определения
# пола индивида, не имеющего детей с помощью утверждений указанных вариантом №4
# курсового проекта (father(father, child)., mother(mother, child).).

# Т.о. на основе таких фактов программа на Прологе сможет и определить пол
# любого из индивидов, и путем введения дополнительного правила (См. rules.pl)
# определить действительную отцовскую\материнскую связь.

for fk in fams.keys():
	f = fams[fk]
	if len(f[2]) == 0:
		output.append("ifather('" + persons[f[0]][0] + "', '" + "nobody" + "').\n")
		output.append("imother('" + persons[f[1]][0] + "', '" + "nobody" + "').\n")  
	else:
		for ch in f[2]:
			child = persons[ch]
			output.append("ifather('" + persons[f[0]][0]+ "', '" + child[0] + "').\n")
			output.append("imother('" + persons[f[1]][0]+ "', '" + child[0] + "').\n")
			if child[2] == 0: 
				if child[1] == "M":
					output.append("ifather('" + child[0]+ "', '" + "nobody" + "').\n")
				else:
					output.append("imother('" + child[0]+ "', '" + "nobody" + "').\n")  
# ---если male\female---
# for k in persons.keys():
# 	p = persons[k]
# 	if p[1] == 'M':
# 		output.append("male('" + p[0] + "').\n")
# 	else:
# 		output.append("female('" + p[0] + "').\n")
#----------------------------------------
outfile = open(outp, "w")

outfile.write("%--DATABASE--\n")
# сортируем т.к. Prolog предпочитает, чтобы однотипные утверждения шли по-порядку
for el in sorted(output):
	outfile.write(el)
outfile.close()