pub struct Reader {
    data: Vec<u8>,
    pos: usize,
}

impl Reader {
    fn from_file(path: &str) -> Self {
        let data = std::fs::read(path).expect("Failed to read file");
        Self { data, pos: 0 }
    }

    pub fn read_i32(&mut self) -> Result<i32, Error> {
        let mut buffer = [0; 4];
        buffer.copy_from_slice(
            self.data
                .get(self.pos..self.pos + 4)
                .ok_or(Error::InvalidPosition(self.pos))?,
        );
        self.pos = self.pos.checked_add(core::mem::size_of::<i32>())
                .ok_or(Error::AddOverflow(self.pos))?;
        Ok(i32::from_le_bytes(buffer))
    }

    pub fn read_u32(&mut self) -> Result<u32, Error> {
        let mut buffer = [0; 4];
        buffer.copy_from_slice(
            self.data
                .get(self.pos..self.pos + 4)
                .ok_or(Error::InvalidPosition(self.pos))?
        );
        self.pos = self.pos.checked_add(core::mem::size_of::<u32>())
                .ok_or(Error::AddOverflow(self.pos))?;
        Ok(u32::from_le_bytes(buffer))
    }

    pub fn read_f32(&mut self) -> Result<f32, Error> {
        Ok(f32::from_le_bytes(
            self.data
                .get(self.pos..self.pos + 4)
                .ok_or(Error::InvalidPosition(self.pos))?
                .try_into().unwrap()
        ))
    }
}

pub enum Error {
    InvalidPosition(usize),
    AddOverflow(usize),
}
