### In your home directory (default when starting cluster) do the following:
```bash
mkdir JupLab
cd ./JupLab
mkdir -p mha/mha
vi mha.yml
vi mha_service.yml
```

- Copy paste the respective contents into the .yml files
- **You'll have to change file path in mha.yml**
- **volumes:hostPath:path will end with your username**

### Run the following commands in order after logging in to cluster
```bash
kubectl create -f mha.yml
kubectl create -f mha_services.yaml
kubectl get services
```

You should see something like
```
NAME              TYPE       CLUSTER-IP       EXTERNAL-IP   PORT(S)        AGE
pytorch-service   NodePort   10.100.162.207   <none>        80:<PORT>/TCP   11m
```
<PORT > is specified by the only commented line in `mha_service.yml`
Go to web browser and type <IP ADDRESS (in your email)>:<PORT > and type in the password `iiitd@123

Run below for setup

```python
!git clone https://github.com/Snehgal/ComplexNeuralNetworks.git
!mv ComplexNeuralNetworks/* .
!pip install -r requirements.txt
```
