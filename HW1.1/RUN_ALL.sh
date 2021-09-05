ScriptLoc=${PWD}

cd LectureCodes

for i in *.py
do
      echo "-----------" $i "-----------" 
      python $i  
      #run all python scripts
      
done

grep "I HAVE WORKED" *

cd $ScriptLoc

for i in *.py
do
      echo "-----------" $i "-----------" 
      python $i  
      #run all python scripts
done

