 
 #Function that takes a parent node and progresses recursively through a tree drawing samples for each child 
 def draw(self, parent:int, sample:list):
        
        children = np.argwhere(self.tree == parent)
        
        for child in children:
            random = np.random.rand()
            if random <= np.exp(self.pmfs)[parent][sample[parent]][0]:
                sample[child[0]] = 1
            self.draw(parent=child[0], sample=sample)
        
        return sample
#Function that draws an amount of i.i.d samples
def sample(self, nsamples: int):
        samples = []

        for i in range(nsamples):
            sample = [0 for i in range(len(self.tree))]
            #Initialize for the root node
            random = np.random.rand()
            
            if random <= np.exp(self.pmfs)[self.root][0][0]:
                sample[self.root] = 1 
            parent = self.root
            
            self.draw(parent=parent, sample=sample)
            samples.append(sample)
        return np.array(samples)