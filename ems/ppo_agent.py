import numpy as np, tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class PPOAgent:
    def __init__(self, action_dim=3, state_dim=3, gamma=0.98, lam=0.95, clip=0.2, lr=1e-3, epochs=5, batch=256):
        self.gamma=gamma; self.lam=lam; self.clip=clip; self.epochs=epochs; self.batch=batch; self.action_dim=action_dim
        inp = keras.Input((state_dim,))
        x = layers.Dense(64, activation='tanh')(inp)
        x = layers.Dense(64, activation='tanh')(x)
        logits = layers.Dense(action_dim)(x)
        value = layers.Dense(1)(x)
        self.model = keras.Model(inp, [logits, value])
        self.opt = keras.optimizers.Adam(lr)
    def act(self, s):
        logits, v = self.model.predict(s[None], verbose=0)
        probs = tf.nn.softmax(logits[0]).numpy()
        a = np.random.choice(self.action_dim, p=probs)
        logp = np.log(probs[a]+1e-8)
        return int(a), float(logp), float(v[0,0])
    def _gae(self, rewards, values, dones, gamma, lam):
        adv = np.zeros_like(rewards); lastgaelam = 0
        for t in reversed(range(len(rewards))):
            nextnonterminal = 1.0 - dones[t]
            nextvalue = values[t+1] if t+1<len(values) else 0.0
            delta = rewards[t] + gamma*nextvalue*nextnonterminal - values[t]
            adv[t] = lastgaelam = delta + gamma*lam*nextnonterminal*lastgaelam
        returns = adv + values[:len(adv)]
        return adv, returns
    def learn(self, buf):
        S = np.array(buf['s'], dtype='float32')
        A = np.array(buf['a'], dtype='int32')
        R = np.array(buf['r'], dtype='float32')
        D = np.array(buf['d'], dtype='float32')
        old_logp = np.array(buf['logp'], dtype='float32')
        logits, V = self.model.predict(S, verbose=0); V = V.squeeze().astype('float32')
        adv, ret = self._gae(R, V, D, self.gamma, self.lam)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ds = tf.data.Dataset.from_tensor_slices((S, A, old_logp, adv, ret)).shuffle(len(S)).batch(self.batch)
        for _ in range(self.epochs):
            for s,a,olp,ad,rt in ds:
                with tf.GradientTape() as tape:
                    logits, v = self.model(s, training=True)
                    logp_all = tf.nn.log_softmax(logits, axis=-1)
                    logp = tf.gather(logp_all, a, batch_dims=1)
                    ratio = tf.exp(logp - olp)
                    pg1 = ratio * ad
                    pg2 = tf.clip_by_value(ratio, 1.0-self.clip, 1.0+self.clip) * ad
                    pg_loss = -tf.reduce_mean(tf.minimum(pg1, pg2))
                    v_loss = tf.reduce_mean((rt - tf.squeeze(v,axis=-1))**2)
                    ent = tf.reduce_mean(-tf.reduce_sum(tf.exp(logp_all)*logp_all, axis=-1))
                    loss = pg_loss + 0.5*v_loss - 0.01*ent
                grads = tape.gradient(loss, self.model.trainable_variables)
                self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
