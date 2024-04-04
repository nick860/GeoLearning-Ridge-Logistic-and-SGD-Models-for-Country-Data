from models import *
from helpers import *

def Stochastic_gradient_descent(train_x, train_y, valid_x, valid_y, test_x, test_y, learning_rate, class_num, n_steps,decay, lambda_value=0, bonus=False):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train = torch.from_numpy(train_x).float().to(device)
    Y_train = torch.from_numpy(train_y).long().to(device)
    X_valid = torch.from_numpy(valid_x).float().to(device)
    Y_valid = torch.from_numpy(valid_y).long().to(device)
    X_test = torch.from_numpy(test_x).float().to(device)
    Y_test = torch.from_numpy(test_y).long().to(device)
    data_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, Y_train), batch_size=32, shuffle=True) 
    save_model = []
    loss_values = []
    ep_correct_preds_train = 0.
    max_valid = -1
    max_valid_index = 0
    if bonus:
        valid_accuracy = {learning_rate[0]: {'losses':[], 'accuracies':[]}}
        test_accuracy = {learning_rate[0]: {'losses':[], 'accuracies':[]}}
        ep_loss_values = {learning_rate[0]: {'losses':[], 'accuracies':[]}}
    else:
        valid_accuracy = {learning_rate[0] : {'losses':[], 'accuracies':[]}, learning_rate[1] : {'losses':[], 'accuracies':[]}, learning_rate[2] : {'losses':[], 'accuracies':[]}}
        test_accuracy = {learning_rate[0] : {'losses':[], 'accuracies':[]}, learning_rate[1] : {'losses':[], 'accuracies':[]}, learning_rate[2] : {'losses':[], 'accuracies':[]}}
        ep_loss_values = {learning_rate[0] : {'losses':[], 'accuracies':[]}, learning_rate[1] : {'losses':[], 'accuracies':[]}, learning_rate[2] : {'losses':[], 'accuracies':[]}}

    for lr in learning_rate:
        model = Logistic_Regression(2, class_num)
        model.to(device)
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr) 
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)
        model.train()
        ep_accuracy = []  
        for epoch in range(n_steps):
            ep_correct_preds_train = 0. 
            loss_values = [] 
            for batch in data_loader:
                x, y = batch
                optimizer.zero_grad() 
                output = model.forward(x) 
                loss = loss_function(output.squeeze(), y) 
                total_loss = loss + lambda_value * torch.sum(model.linear.weight**2)
                total_loss.backward()
                optimizer.step()
                loss_values.append(total_loss.item())
                ep_correct_preds_train += torch.sum(torch.argmax(output, dim=1) == y).item()

            mean_loss = np.mean(loss_values)
            ep_accuracy = ep_correct_preds_train / len(train_y)          
            ep_loss_values[lr]['accuracies'].append(ep_accuracy)
            ep_loss_values[lr]['losses'].append(mean_loss)

            with torch.no_grad():
                model.eval()
                output = model.forward(X_valid)
                loss = loss_function(output.squeeze(), Y_valid)
                valid_accuracy[lr]['accuracies'].append(torch.sum(torch.argmax(output, dim=1) == Y_valid).item() / len(Y_valid))
                valid_accuracy[lr]['losses'].append(loss.item())

                output = model.forward(X_test)
                loss = loss_function(output.squeeze(), Y_test)
                test_accuracy[lr]['accuracies'].append(torch.sum(torch.argmax(output, dim=1) == Y_test).item() / len(Y_test))
                test_accuracy[lr]['losses'].append(loss.item())
                
                if decay:
                    lr_scheduler.step()
        if valid_accuracy[lr]['accuracies'][-1] > max_valid:
            max_valid = valid_accuracy[lr]['accuracies'][-1]
            max_valid_index = learning_rate.index(lr)
            the_best_model_valid = model
        save_model.append(model)

    if not bonus:
        if (decay):
            plot_accuracy_vs_learning_rate(learning_rate, [valid_accuracy[lr]['accuracies'][-1] for lr in learning_rate], [test_accuracy[lr]['accuracies'][-1] for lr in learning_rate])
            plot_accuracy_vs_epoch(ep_loss_values[lr]['accuracies'], valid_accuracy[lr]['accuracies'], test_accuracy[lr]['accuracies'], n_steps)
        else:
            helpers.plot_decision_boundaries(the_best_model_valid, test_x, test_y, title="learning rate = " + str(learning_rate[max_valid_index]))
        lr = learning_rate[max_valid_index]
        print("his validation accuracy " + str(valid_accuracy[lr]['accuracies'][-1]))
        print("his test accuracy " + str(test_accuracy[lr]['accuracies'][-1]))
        plot_loss_vs_epoch(ep_loss_values[lr]['losses'], valid_accuracy[lr]['losses'], test_accuracy[lr]['losses'], n_steps)
        
            
    return [valid_accuracy[lr]['accuracies'][-1],test_accuracy[lr]['accuracies'][-1]], the_best_model_valid

def plot_loss_vs_epoch(train_loss,valid_loss, test_loss, n_steps):
    plt.scatter(range(n_steps), train_loss, label='Train')
    plt.scatter(range(n_steps), valid_loss, label='Validation')
    plt.scatter(range(n_steps), test_loss, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs Epoch')
    plt.legend()
    plt.show()

def plot_accuracy_vs_epoch(train_accuracy, valid_accuracy, test_accuracy, n_steps):
    plt.scatter(range(n_steps), train_accuracy, label='Train')
    plt.scatter(range(n_steps), valid_accuracy, label='Validation')
    plt.scatter(range(n_steps), test_accuracy, label='Test')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Epoch')
    plt.legend()
    plt.show()

def plot_accuracy_vs_learning_rate(learning_rate, valid_accuracy, test_accuracy):
    plt.scatter(learning_rate, valid_accuracy, label='Validation')
    plt.scatter(learning_rate, test_accuracy, label='Test')
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Learning Rate')
    plt.legend()
    plt.show()
        
def df_dx(x):
    return 2*(x-3)
def df_dy(y):
    return 2*(y-5)
def gradient_descent_q1 (n_steps, learning_rate):
    x,y= 0, 0
    # for the function = (x-3)**2 + (y-5)**2
    x1 = [0]
    y1 = [0]
    for i in range(n_steps):
        x = x - learning_rate * df_dx(x)
        y = y - learning_rate * df_dy(y)
        x1.append(x)
        y1.append(y)

    plt.scatter(x1, y1, c=range(n_steps+1), cmap='viridis')    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gradient Descent')
    plt.colorbar(label = 'step')
    plt.show()


def ridge_regression_q1(X_train, Y_train, X_valid, Y_valid, X_test, Y_test):
    y_train = []
    y_valid = []
    y_test = []
    save_model = []
    for alpha in range(0, 11, 2):
        ridge_model = Ridge_Regression(alpha)  # Move the initialization inside the loop
        ridge_model.fit(X_train, Y_train)
        save_model.append(ridge_model)
        y_train.append(np.mean(ridge_model.predict(X_train) == Y_train))
        y_valid.append(np.mean(ridge_model.predict(X_valid) == Y_valid))
        y_test.append(np.mean(ridge_model.predict(X_test) == Y_test))

    plt.scatter(range(0, 11, 2), y_train, label='Train')
    plt.scatter(range(0, 11, 2), y_valid, label='Validation')
    plt.scatter(range(0, 11, 2), y_test, label='Test')
    plt.xlabel('λ')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs λ')
    plt.legend()
    plt.show()
    best_model_index = np.argmax(y_valid)
    worst_model_index = np.argmin(y_valid)
    lambda_best = best_model_index * 2
    lambda_worst = worst_model_index * 2
    print("the best model is the one with lambda = " + str(lambda_best))
    print("the worst model is the one with lambda = " + str(lambda_worst))
    print("the test accurcay of the best model "+ str(y_test[best_model_index]))
    print("the test accurcay of the worst model "+ str(y_test[worst_model_index]))
    helpers.plot_decision_boundaries(save_model[best_model_index], X_test, Y_test, title="lambda = " + str(lambda_best))
    helpers.plot_decision_boundaries(save_model[worst_model_index],  X_test, Y_test, title="lambda = " + str(lambda_worst))
    return y_train, y_valid, y_test

def decision_tree(X_train, Y_train, X_test, Y_test, max_depth):

    decision_tree = DecisionTreeClassifier(max_depth=max_depth)
    decision_tree.fit(X_train, Y_train)
    print("the accuracy of the decision tree is " + str(np.mean(decision_tree.predict(X_test) == Y_test)))
    helpers.plot_decision_boundaries(decision_tree, X_test, Y_test, title="Decision Tree with depth " + str(max_depth))

def bonus(X_train, Y_train, X_valid, Y_valid, X_test, Y_test):
    best_model = None
    best_accuracy_valid = 0
    best_accuracy_test = 0
    best_lambda = 0
    for i in range(0, 11, 2):
        print("lambda = " + str(i))
        accuracy , model = Stochastic_gradient_descent(X_train, Y_train, X_valid, Y_valid, X_test, Y_test, [0.01], 5, 30, True, i, True)
        if accuracy[0] > best_accuracy_valid:
            best_model = model
            best_accuracy_valid = accuracy[0]
            best_accuracy_test = accuracy[1]
            best_lambda = i
        print("his valid accuracy is " + str(accuracy[0]))
        print("his test accuracy is " + str(accuracy[1]))

    helpers.plot_decision_boundaries(best_model, X_test, Y_test, title="lambda = " + str(best_lambda))
    print("his accuracy is " + str(best_accuracy_test))

if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    # Load the train data
    train_data, col_names = helpers.read_data_demo('train.csv')
    valid_data, valid_col = helpers.read_data_demo('validation.csv')
    test_data, test_col = helpers.read_data_demo('test.csv')
    train_data_multi, col_names_multi = helpers.read_data_demo('train_multiclass.csv')
    valid_data_multi, valid_col_multi = helpers.read_data_demo('validation_multiclass.csv')
    test_data_multi, test_col_multi = helpers.read_data_demo('test_multiclass.csv')
    # Split the data into features (X) and labels (Y)
    X_train = train_data[:, :-1]
    Y_train = train_data[:, 2]

    X_valid = valid_data[:, :-1]
    Y_valid = valid_data[:, 2]

    X_test = test_data[:, :-1]
    Y_test = test_data[:, 2]

    X_train_multi = train_data_multi[:, :-1]
    Y_train_multi = train_data_multi[:, 2]

    X_valid_multi = valid_data_multi[:, :-1]
    Y_valid_multi = valid_data_multi[:, 2]

    X_test_multi = test_data_multi[:, :-1]
    Y_test_multi = test_data_multi[:, 2]
    learning_rate = [0.1, 0.01, 0.001]
    learning_rate_multi = [0.01, 0.001, 0.0003]
    ridge_regression_q1(X_train, Y_train, X_valid, Y_valid, X_test, Y_test)
    gradient_descent_q1(1000, 0.1)
    Stochastic_gradient_descent(X_train, Y_train, X_valid, Y_valid, X_test, Y_test, learning_rate, 2, 10, False)
    Stochastic_gradient_descent(X_train_multi, Y_train_multi, X_valid_multi, Y_valid_multi, X_test_multi, Y_test_multi, learning_rate_multi, 5, 30, True)
    decision_tree(X_train_multi, Y_train_multi, X_test_multi, Y_test_multi, 2)
    decision_tree(X_train_multi, Y_train_multi, X_test_multi, Y_test_multi, 10)
    bonus(X_train_multi, Y_train_multi, X_valid_multi, Y_valid_multi, X_test_multi, Y_test_multi)
