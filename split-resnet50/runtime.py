from typing_extensions import Self

class RuntimeRecord:
    def __init__(self, category_name: str, parent_category: Self = None):
        self.category_name = category_name
        self.parent_category = parent_category
        self.total_runtime = 0.0
        self.subcategories = []

    def create_subcategory(self, subcategory_name: str) -> Self:
        subcategory = RuntimeRecord(subcategory_name, parent_category = self)
        self.subcategories.append(subcategory)
        return subcategory
    
    def net_runtime_per_category(self, category_name: str) -> float:
        net_runtime = 0.0
        if category_name == self.category_name:
            net_runtime += self.total_runtime
        
        for subcategory in self.subcategories:
            net_runtime += subcategory.net_runtime_per_category(category_name)

        return net_runtime
    
    def print_runtime(self, indent_level: int = 0):
        if self.parent_category == None:
            runtime_percentage = 100
        else:
            runtime_percentage = self.total_runtime / self.parent_category.total_runtime * 100
        
        print('{}- {}: {:.2f}% ({:.7f})'.format(' '* indent_level * 4, self.category_name, runtime_percentage, self.total_runtime))
        for subcategory in self.subcategories:
            subcategory.print_runtime(indent_level + 1)


if __name__ == '__main__':
    record = RuntimeRecord('Resnet')
    record.total_runtime = 100
    part1 = record.create_subcategory('part1')
    part1.total_runtime = 70
    part1_conv = part1.create_subcategory('conv')
    part1_conv.total_runtime = 30
    part1_concat = part1.create_subcategory('concat')
    part1_concat.total_runtime = 40
    part2 = record.create_subcategory('part2')
    part2.total_runtime = 30
    part2_conv = part2.create_subcategory('conv')
    part2_conv.total_runtime = 15
    part2_concat = part2.create_subcategory('concat')
    part2_concat.total_runtime = 15

    assert(record.net_runtime_per_category('conv') == (part1_conv.total_runtime + part2_conv.total_runtime))
    assert(record.net_runtime_per_category('concat') == (part1_concat.total_runtime + part2_concat.total_runtime))

    record.print_runtime()
