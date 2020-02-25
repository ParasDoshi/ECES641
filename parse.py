import csv

class CSVService:
    """A class to handle CSV parsing service"""
    metabolitePath = ""
    abundancePath = ""
    
    def __init__(self, metabolitePath, abundancePath):
        self.metabolitePath = metabolitePath
        self.abundancePath = abundancePath
    
    def CleanNames(self, names):
        """CleanNames accepts a list of names from the CSV and removes the `s__` prefix
        and replaces any white space with an underscore.
        Example: s__Sulfolobus islandicus rod-shaped virus 1 -> Sulfolobus_islandicus_rod-shaped_virus_1
        """
        for i in range(0, len(names)):
            if names[i].startswith('s__'):
                names[i] = names[i][len('s__'):]
            names[i] = names[i].replace(" ", "_")
        return names

    def ParseAbundance(self):
        """ParseAbundance parses the abundance csv file
        and returns a tuple where the first element is a
        list of species names and the second element is a
        matrix of concentrations."""
        species = []
        with open(self.abundancePath, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            rows = 0
            concentrations = []
            for row in reader:
                if rows == 0:
                    # The first row is just a listing of the species, so lets keep track of them.
                    species = row
                    rows += 1
                    continue
                concentrations.append(row)
        return (species, concentrations)        

    def ParseMetabolite(self):
        """ParseMetabolite parses the metabolite csv file
        and returns a dictionary where the key is the microbe
        and the value is a list of concentrations."""
        species = []
        with open(self.metabolitePath, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            rows = 0
            concentrations = []
            for row in reader:
                if rows == 0:
                    # The first row is just a listing of the metabolites, so lets keep track of them.
                    metabolites = row
                    rows += 1
                    continue
                concentrations.append(row)
        return (metabolites, concentrations)  

class CorpusService:
    "A class to handle creating and preparing the corpus file"
    species = []
    metabolites = []
    speciesAbundance = []
    metaboliteConcentration = []
    filepath = ""

    def __init__(self, species, metabolites, speciesAbundance, metaboliteConcentration, filepath):
        self.species = species
        self.metabolites = metabolites
        self.speciesAbundance = speciesAbundance
        self.metaboliteConcentration = metaboliteConcentration
        self.filepath = filepath

    def CreateCorpusFile(self, abundanceMatrix, concentrationMatrix, species, metabolites, threshold):
        """CreateCorpusFile accepts a matrix of abundances and a matrix of concentrations as well as the 
        metabolite and species names. It then iterates row wise over the concentrations/abundances. While iterating
        through each column, if the current value is greater than threshold, then the corresponding name gets printed to the outfile."""
        f = open(self.filepath, 'w')
        if len(abundanceMatrix) != len(concentrationMatrix):
            print('matrix sizes do not match.')
            return
        for i in range(0, len(abundanceMatrix)):
            somethingWritten = False
            aRow = abundanceMatrix[i]
            for j in range(0, len(aRow)):
                if float(aRow[j]) > threshold:
                    somethingWritten = True
                    f.write("{} ".format(species[j]))
            cRow = concentrationMatrix[i]
            for j in range(0, len(cRow)):
                if float(cRow[j]) > threshold:
                    somethingWritten = True
                    f.write("{} ".format(metabolites[j]))
            if somethingWritten:
                f.write("\n")
        f.close()

    def CreateCorpusDict(self, names, values):
        corpus = {}
        if len(names) != len(values):
            return corpus
        for i in range(0, len(names)):
            corpus[names[i]] = values[i]
        return corpus

    def CreateAverageCorpus(self):
        """CreateAverageCorpus accepts a list of species names, metabolite names
        species concentration, and metabolite concentrations and returns a tuple
        where the first element is the average abundances and the second element
        is the average concentrations"""
        averageAbundances = [None] * len(self.speciesAbundance[0])
        for i in range(0, len(self.speciesAbundance)):
            row = self.speciesAbundance[i]
            for j in range(0,len(row)):
                if i == 0:
                    averageAbundances[j] = float(row[j])
                    continue
                averageAbundances[j] = (averageAbundances[j] + float(row[j])) / 2

        averageConcentrations = [None] * len(self.metaboliteConcentration[0])
        for i in range(0, len(self.metaboliteConcentration)):
            row = self.metaboliteConcentration[i]
            for j in range(0,len(row)):
                if i == 0:
                    averageConcentrations[j] = float(row[j])
                    continue
                averageConcentrations[j] = (averageConcentrations[j] + float(row[j])) / 2
        return (averageAbundances, averageConcentrations)

if __name__ == "__main__":
    csvService = CSVService('metabolite_table.csv', 'abundance_table.csv')
    species, abundances = csvService.ParseAbundance()
    species.pop(0) # this is an empty column
    species.pop(0) # this is `sample_id` column header
    species.pop(0) # this is `label` column header
    species.pop(-1) # this is `run_id` column header
    
    for i in range(0, len(abundances)):
        abundances[i].pop(0) # this is the row index
        abundances[i].pop(0) # this is the `sample_id`
        abundances[i].pop(0) # this is the `label`
        abundances[i].pop(-1) # this is the `run_id`

        if len(species) != len(abundances[i]):
            print('error with abundance matrix. Sizes did not match')

    metabolites, concentrations = csvService.ParseMetabolite()
    metabolites.pop(0) # this is an empty column
    metabolites.pop(0) # this is the `factors` column header
    metabolites.pop(-1) # this is the `label` column header
    for i in range(0, len(concentrations)):
        concentrations[i].pop(0) # this is an sample_id
        concentrations[i].pop(0) # this is the `factor` 
        concentrations[i].pop(-1) # this is the `label`  
        if len(metabolites) != len(concentrations[i]):
            print('error with concentration matrix. Sizes did not match')


    species = csvService.CleanNames(species)
    metabolites = csvService.CleanNames(metabolites) 

    corpusService = CorpusService(species, metabolites, abundances, concentrations, "corpus.txt")
    averageAbundances, averageConcentrations = corpusService.CreateAverageCorpus()
    # abundanceMatrix, concentrationMatrix, species, metabolites, threshold)
    corpusService.CreateCorpusFile(abundances, concentrations, species, metabolites, 0)
