PATH=C:\tools\MongoDB\Server\4.2\bin:%PATH%
rem mongoexport --uri='mongodb://localhost:27017/ConstructionDB' --collection=scoring_factory --out=scoring_factory.json

rem C:\tools\MongoDB\Server\4.2\bin\mongoexport.exe  --uri="mongodb://localhost:27017/ConstructionDB" --collection=scoring_factory --out=export/scoring_factory.json

rem C:\tools\MongoDB\Server\4.2\bin\mongoexport.exe  --uri="mongodb://localhost:27017/ConstructionDB" --collection=metrics --out=export/metrics.json --limit=1000000


C:\tools\MongoDB\Server\4.2\bin\mongoexport.exe  --uri="mongodb://localhost:27017/ConstructionDB" --collection=sessions --out=export/sessions.json --limit=10000000
C:\tools\MongoDB\Server\4.2\bin\mongoexport.exe  --uri="mongodb://localhost:27017/ConstructionDB" --collection=sessionsV2 --out=export/sessionsV2.json --limit=10000000
C:\tools\MongoDB\Server\4.2\bin\mongoexport.exe  --uri="mongodb://localhost:27017/ConstructionDB" --collection=users --out=export/users.json --limit=10000000
C:\tools\MongoDB\Server\4.2\bin\mongoexport.exe  --uri="mongodb://localhost:27017/ConstructionDB" --collection=classes --out=export/classes.json --limit=10000000

