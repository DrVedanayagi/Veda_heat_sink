students_count=1000
print(students_count)
course = "python name sample"
print(len(course))
first = "mosh"
last = "hamedani"
full = first + " " + last
print(full)
#striings
course = "python for beginners"
print(course.upper())
print(course.lower())
print(course.find("for"))
print(course.replace("beginners", "absolute beginners"))
print(course.title())

print("python" in course) #this is an expression , it a piece of code that produces a value
print("python" not in course)
print("swift" in course)
#conditional statements
temperature = 5
if temperature > 30:
    print("it's a hot day")
    print("drink plenty of water")
elif temperature > 20:
    print("it's a nice day")
else:
    print("it's cold day")
print("done")
#for else
successful= True
for number in range (3):
    print("attempt", number + 1)
    if successful:
        print("successful")
        break
else:
    print("attempt failed")
    
successful= False
for number in range (3):
    print("attempt", number + 1)
    if successful:
        print("successful")
        break
else:
    print("attempt failed")

#while loop
number = 100
while number > 0:
    print(number)
    number //= 2

numbers = 10
while numbers > 0:
    print (numbers)
    numbers -=2
print("we have 4 even numbers")

range(1,10)
for number in range(2,10,2):
    print(number) 
print("we have 4 even numbers")

#using count
count = 0 
for number in range(1, 10):
    if number % 2 ==0:
        count += 1
        print(number)
print(f"we have {count} even numbers")
